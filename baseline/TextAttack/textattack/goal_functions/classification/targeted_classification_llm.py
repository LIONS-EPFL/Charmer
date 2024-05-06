"""

Determine if an attack has been successful in targeted Classification
-----------------------------------------------------------------------
"""

# TODO, how to deal with weird output...
from textattack.goal_functions import GoalFunction
from textattack.goal_function_results.goal_function_result import (
    GoalFunctionResultStatus,
)
from textattack.shared.utils import ReprMixin
from textattack.goal_function_results import ClassificationGoalFunctionResult
import pdb

class TargetedClassificationLLM(GoalFunction):

    """A goal function defined on a model that outputs a probability for some
    number of classes."""

    def __init__(self, inference, query_budget,verbose=True, logger=None, *args,**kwargs):
        self.inference = inference
        self.query_budget = query_budget
        self.logger = logger
        self.verbose = verbose
        super().__init__(*args, **kwargs)
        
    def _process_model_outputs(self, inputs, outputs):
        return outputs

    def _is_goal_complete(self, model_output, _):
        return self.target_class == model_output.argmax()#.item()
        
    def _get_score(self, model_output, _):
        return model_output[self.target_class]
        # if self.target_class == model_output:
        #     score = 1
        # else:
        #     score = 0
        # if self.original_class != model_output:
        #     score = 1
        # else:
        #     score = 0
        # return score

    def _goal_function_result_type(self):
        """Returns the class of this goal function's results."""
        return ClassificationGoalFunctionResult

    def extra_repr_keys(self):
        return []

    def _get_displayed_output(self, raw_output):
        return raw_output
    
    def _call_model(self, text_list):
        if len(text_list[0]._text_input.values())>1: # (premise,orig_S)
            text_list = [list(i._text_input.values()) for i in text_list]
        else:
            text_list = [i.text for i in text_list]
        pred_list = self.inference(text_list) 
        if pred_list[0].is_cuda:
            return [i.detach().cpu() for i in pred_list]
        else:
            return pred_list

    def _get_goal_status(self, model_output, text, check_skip=False):
        return super()._get_goal_status(model_output, text, check_skip)


