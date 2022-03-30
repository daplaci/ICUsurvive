#!/bin/bash

python -u generate_file.py --debug 0 --action map_bcc
python -u generate_file.py --debug 0 --action map_labka
python -u generate_file.py --debug 0 --action dump_from_pickle
