## Copyright 2024 Kaden Bilyeu (Bikatr7) (https://github.com/Bikatr7), Alejandro Mata (https://github.com/alemalvarez)
## EasyTL (https://github.com/Bikatr7/EasyTL)
## Use of this source code is governed by an GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

## built-in imports
import typing

class LDHelper:

    """
    
    Logging Directory helper so I don't lose my mind trying to make the damn thing work in Elucidate

    """

    current_logging_directory:str | None = None
    current_cls_name:str

    @staticmethod
    def set_logging_directory_attributes(directory:str | None, cls_name:str) -> None:

        """
        
        Parameters:
        directory (string) : The directory to set as the logging directory.
        cls_name (string) : The name of the class to set as the current class

        """

        LDHelper.current_logging_directory = directory
        LDHelper.current_cls_name = cls_name

    
    @staticmethod
    def get_logging_directory_attributes() -> typing.Tuple[str | None, str]:

        """
        
        Returns:
        string : The current logging directory.
        string : The current class name
        
        """

        return LDHelper.current_logging_directory, LDHelper.current_cls_name
    