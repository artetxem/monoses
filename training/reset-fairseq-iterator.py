# Copyright (C) 2019  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import torch


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Reset the iterator of a fairseq model')
    parser.add_argument('model', help='path to the model')
    args = parser.parse_args()

    state = torch.load(args.model)
    state['extra_state']['train_iterator']['epoch'] = 0
    state['extra_state']['train_iterator']['iterations_in_epoch'] = 0
    torch.save(state, args.model)


if __name__ == '__main__':
    main()
