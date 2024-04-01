## Copyright Bikatr7 (https://github.com/Bikatr7)
## Use of this source code is governed by a GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

## deepL generic exception
from deepl.exceptions import DeepLException

## google generic exception
from google.api_core.exceptions import GoogleAPIError

##-------------------start-of-InvalidAPIKeyException--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class InvalidAPIKeyException(Exception):

    """

    InvalidAPIKeyException is an exception that is raised when the API key is invalid.

    """

    def __init__(self, model_name:str) -> None:

        """

        Parameters:
        model_name (string) : The name of the model that the API key is invalid for.

        """

        self.message = f"The API key is invalid for the model {model_name}."