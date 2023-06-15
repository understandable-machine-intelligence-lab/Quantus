# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.
import subprocess

commit_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
version = subprocess.check_output(['git', 'describe', '--tags', '--always', '--dirty=-pre', commit_sha]).strip().decode('utf-8')

__version__ = version

# Expose quantus.evaluate to the user.
from quantus.evaluation import evaluate

# Expose quantus.explain to the user.
from quantus.functions.explanation_func import explain

# Expose quantus.<function-class>.<function-name> to the user.
from quantus.functions import *

# Expose quantus.<metric> to the user.
from quantus.metrics import *

# Expose quantus.helpers.constants to the user.
from quantus.helpers.constants import *
