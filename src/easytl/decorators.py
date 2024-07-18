## Copyright 2024 Kaden Bilyeu (Bikatr7) (https://github.com/Bikatr7), Alejandro Mata (https://github.com/alemalvarez)
## EasyTL (https://github.com/Bikatr7/EasyTL)
## Use of this source code is governed by an GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

## One day this absolute disaster of a file will break and I will have to fix it. But today is not that day.

## built-in libraries
from functools import wraps

import datetime
import os
import logging

## custom modules
from .classes import AnthropicTextBlock, AnthropicToolUseBlock

##-------------------start-of-get_nested_attribute()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _get_nested_attribute(obj, attrs):
    for attr, type_hint in attrs:
        try:
            if(isinstance(obj, list) and attr.isdigit()):
                obj = obj[int(attr)]
            else:
                try:
                    if(type_hint is None or isinstance(obj, type_hint)):
                        obj = getattr(obj, attr)
                except AttributeError:
                    ## Try dictionary access
                    obj = obj[attr]

        except (AttributeError, IndexError, KeyError):
            raise ValueError(f"Attribute {attr} in object {obj} not found.")
        
    return str(obj)

##-------------------start-of-logging_decorator()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _async_logging_decorator(func):

    @wraps(func)
    async def wrapper(*args, **kwargs):
        ## Get the class name from the function's qualified name
        cls_name = func.__qualname__.split('.')[0]
        
        ## Perform lazy import of the module where the class is defined
        module_name = func.__module__
        module = __import__(module_name, fromlist=[cls_name])
        cls = getattr(module, cls_name)

        if(not hasattr(cls, '_log_directory')):
            logging.warning(f"Class {cls_name} does not have a log directory set. Logging Directory will not function.")

        directory = getattr(cls, '_log_directory', None)
        if(directory is None):
            return await func(*args, **kwargs)

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        filename = f"easytl-log.txt"
        filepath = os.path.join(directory, filename)
        
        result = await func(*args, **kwargs)

        ## Get the attribute to log
        attr_to_logs = log_attributes.get(cls_name, None)
        if(attr_to_logs):
            log_data = []
            for attr_to_log in attr_to_logs:
                if(not isinstance(attr_to_log, list) and cls_name != 'OpenAIService'):
                    ## coerce to list
                    attr_to_log = [attr_to_log]
                if(not isinstance(result, str)):
                    log_data.append(_get_nested_attribute(result, attr_to_log))
            log_data = '\n'.join(log_data)
        
        ## did you know multi-line f-strings take leading spaces into account?
        log_data = f"""
{'=' * 40}
Function Call Details:
{'-' * 40}
Class Name: {cls_name}
Function Name: {func.__name__}
Arguments: {args}
Keyword Arguments: {kwargs}
{'-' * 40}
Result Details:
{'-' * 40}
Result: {log_data}
{'-' * 40}
Timestamp: {timestamp}
{'=' * 40}
        """
        
        with open(filepath, 'a+') as file:
            file.write(log_data)
        
        return result
    
    return wrapper

def _sync_logging_decorator(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        ## Get the class name from the function's qualified name
        cls_name = func.__qualname__.split('.')[0]
        
        ## Perform lazy import of the module where the class is defined
        module_name = func.__module__
        module = __import__(module_name, fromlist=[cls_name])
        cls = getattr(module, cls_name)

        if(not hasattr(cls, '_log_directory')):
            logging.warning(f"Class {cls_name} does not have a log directory set. Logging Directory will not function.")
            
        directory = getattr(cls, '_log_directory', None)
        if(directory is None):
            return func(*args, **kwargs)

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        filename = f"easytl-log.txt"
        filepath = os.path.join(directory, filename)
        
        result = func(*args, **kwargs)

        ## Get the attribute to log
        attr_to_logs = log_attributes.get(cls_name, None)
        if(attr_to_logs):
            log_data = []
            for attr_to_log in attr_to_logs:
                if(not isinstance(attr_to_log, list) and cls_name != 'OpenAIService'):
                    ## coerce to list
                    attr_to_log = [attr_to_log]
                if(not isinstance(result, str)):
                    log_data.append(_get_nested_attribute(result, attr_to_log))
            log_data = '\n'.join(log_data)
        
        ## did you know multi-line f-strings take leading spaces into account?
        log_data = f"""
{'=' * 40}
Function Call Details:
{'-' * 40}
Class Name: {cls_name}
Function Name: {func.__name__}
Arguments: {args}
Keyword Arguments: {kwargs}
{'-' * 40}
Result Details:
{'-' * 40}
Result: {log_data}
{'-' * 40}
Timestamp: {timestamp}
{'=' * 40}
        """
        
        with open(filepath, 'a+') as file:
            file.write(log_data)
        
        return result
    
    return wrapper

## Since we're dealing with objects here...
log_attributes = {
    'GeminiService': [('text', None)],
    'DeepLService': [('text', None)],
    'OpenAIService': [[('choices', None), ('0', None), ('message', None), ('content', None)],
                      [('choices', None), ('0', None), ('message', None), ('content', None)],
                      ],
    'GoogleTLService': [('translatedText', None)],
    'AnthropicService': [
        [('content', None), ('0', None), ('text', AnthropicTextBlock)],
        [('content', None), ('0', None), ('input', AnthropicToolUseBlock)]
    ],
    'AzureService': [
        [('0', None), ('translations', None), ('0', None), ('text', None)]
    ]
}
