## Copyright Bikatr7 (https://github.com/Bikatr7)
## Use of this source code is governed by a GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

from typing import Iterable

def convert_iterable_to_str(iterable:Iterable) -> str:
    return "".join(map(str, iterable))