from __future__ import print_function
from __future__ import absolute_import

import logging

from flask import Flask

from lionlbd.config import DEBUG, TRIM_BLOCKS

if DEBUG:
    logging.getLogger().setLevel(logging.DEBUG)
    
from lionlbd.startup import graph


app = Flask(__name__)
app.jinja_env.trim_blocks = TRIM_BLOCKS


from lionlbd import views
