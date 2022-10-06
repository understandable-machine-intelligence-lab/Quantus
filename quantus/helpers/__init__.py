# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from importlib import util

__EXTRAS__ = util.find_spec("captum") or util.find_spec("tf_explain")
__MODELS__ = util.find_spec("torch") or util.find_spec("tensorflow")

from .asserts import *
from .constants import *
from .norm_func import *
from .normalise_func import *
from .mosaic_func import *
from .loss_func import *
from .discretise_func import *
from .perturb_func import *
from .plotting import *
from .similarity_func import *
from .utils import *
from .warn_func import *

# Import files dependent on package installations.
if __MODELS__:
    from .models import *
if __EXTRAS__:
    from .explanation_func import *
