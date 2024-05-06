class TransformerRunner(ModelRunner):
    def __init__(self, recoverer, output_mode, label_list, output_dir, device, task_name,
                 model_type, model_name_or_path, do_lower_case, max_seq_length):
        super(TransformerRunner, self).__init__(recoverer, output_mode, label_list, output_dir, device)
        self.task_name = task_name
        self.model_type = model_type
        self.do_lower_case = do_lower_case
        self.max_seq_length = max_seq_length
        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        self.model_class = model_class
        self.tokenizer_class = tokenizer_class
        config = config_class.from_pretrained(model_name_or_path, num_labels=len(label_list), 
                                              finetuning_task=task_name)
        self.tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=do_lower_case)
        self.model = model_class.from_pretrained(
                model_name_or_path, from_tf=bool('.ckpt' in model_name_or_path), config=config)
        self.model.to(device)

    def _prep_examples(self, examples, verbose=False):
        features = convert_examples_to_features(
                examples, self.label_list, self.max_seq_length, self.tokenizer, self.output_mode,
                cls_token_at_end=bool(self.model_type in ['xlnet']),  # xlnet has a cls token at the end
                cls_token=self.tokenizer.cls_token,
                sep_token=self.tokenizer.sep_token,
                cls_token_segment_id=2 if self.model_type in ['xlnet'] else 0,
                pad_on_left=bool(self.model_type in ['xlnet']),  # pad on the left for xlnet
                pad_token_segment_id=4 if self.model_type in ['xlnet'] else 0,
                verbose=verbose)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        if self.output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        elif self.output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
        all_text_ids = torch.tensor([f.example_idx for f in features], dtype = torch.long)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_text_ids)
        return dataset

    def train(self, train_data, args):
        print("Preparing examples.")
        train_dataset = self._prep_examples(train_data, verbose=args.verbose)
        print("Starting training.")
        global_step, tr_loss, train_results = train(args, train_dataset, self.model, self.tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        logger.info("Saving model checkpoint to %s", self.output_dir)
        model_to_save = model.module if hasattr(self.model, 'module') else self.model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        torch.save(args, os.path.join(self.output_dir, 'training_args.bin'))

        # Reload model
        self.load(self.output_dir, self.device)
        print("Finished training.")

    def load(self, output_dir, device):
        self.model = self.model_class.from_pretrained(output_dir)
        self.tokenizer = self.tokenizer_class.from_pretrained(output_dir)
        self.model.to(self.device)

    def query(self, examples, batch_size, do_evaluate=True, return_logits=False, 
              do_recover=True, use_tqdm=True):
        if do_recover:
            examples = [self.recoverer.recover_example(x) for x in examples]
        dataset = self._prep_examples(examples) 
        eval_sampler = SequentialSampler(dataset)  # Makes sure order is correct
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size)

        # Eval!
        logger.info("***** Querying model *****")
        logger.info("  Num examples = %d", len(examples))
        logger.info("  Batch size = %d", batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        example_idxs = None
        self.model.eval()
        if use_tqdm:
            eval_dataloader = tqdm(eval_dataloader, desc="Querying")
        for batch in eval_dataloader:
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if self.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                          'labels':         batch[3]}
                outputs = self.model(**inputs)
                inputs['example_idxs'] = batch[4]
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
                example_idxs = inputs['example_idxs'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
                example_idxs = np.append(example_idxs, inputs['example_idxs'].detach().cpu().numpy(), axis = 0)

        eval_loss = eval_loss / nb_eval_steps
        logger.info('  eval_loss = %.6f', eval_loss)
        incorrect_example_indices = None
        if self.output_mode == "classification":
            pred_argmax = np.argmax(preds, axis=1)
            pred_labels = [self.label_list[pred_argmax[i]] for i in range(len(examples))]
            incorrect_example_indices = set(example_idxs[np.not_equal(pred_argmax, out_label_ids)])

        elif self.output_mode == "regression":
            preds = np.squeeze(preds)

        if do_evaluate:
            result = compute_metrics(self.task_name, pred_argmax, out_label_ids)
            output_eval_file = os.path.join(self.output_dir, "eval-{}.txt".format(self.task_name))
            #print("Possible predictions: ", set(list(preds)))
            #priny("Model predictions: mean: {}, max: {}, min: {}".format(preds.mean(), preds.max(), preds.min()))
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
            
        if return_logits:
            return preds
        else:
            return pred_labels
