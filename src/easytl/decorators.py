## Copyright Bikatr7 (https://github.com/Bikatr7)
## Use of this source code is governed by a GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

## built-in libraries
from functools import wraps

import datetime
import os

##-------------------start-of-get_nested_attribute()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _get_nested_attribute(obj, attrs):
    for attr in attrs:
        try:
            if(isinstance(obj, list) and attr.isdigit()):
                obj = obj[int(attr)]
            else:
                try:
                    obj = getattr(obj, attr)
                except AttributeError:
                    ## Try dictionary access
                    obj = obj[attr]

        except (AttributeError, IndexError, KeyError):
            raise ValueError(f"Attribute {attr} in object {obj} not found.")
        
    return obj

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
            raise ValueError(f"No log directory defined for class {cls_name}")

        directory = getattr(cls, '_log_directory', None)
        if(directory is None):
            return await func(*args, **kwargs)

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        filename = f"easytl-log.txt"
        filepath = os.path.join(directory, filename)
        
        result = await func(*args, **kwargs)

        ## Get the attribute to log
        attr_to_log = log_attributes.get(cls_name, None)
        if(attr_to_log and not isinstance(result, str)):
            log_data = _get_nested_attribute(result, attr_to_log)
        
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
            raise ValueError(f"No log directory defined for class {cls_name}")

        directory = getattr(cls, '_log_directory', None)
        if(directory is None):
            return func(*args, **kwargs)

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        filename = f"easytl-log.txt"
        filepath = os.path.join(directory, filename)
        
        result = func(*args, **kwargs)

        ## Get the attribute to log
        attr_to_log = log_attributes.get(cls_name, None)
        if(attr_to_log and not isinstance(result, str)):
            log_data = _get_nested_attribute(result, attr_to_log)
        
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
    'GeminiService': ['text'],
    'DeepLService': ['text'],
    'OpenAIService': ['choices', '0', 'message', 'content'],
    'GoogleTLService': ['translatedText']
}