"""Usage:
  mlcli tfrecord <path>
"""

from docopt import docopt
from ml.cli import tfrecord


def main():
    args = docopt(__doc__)
    if args['tfrecord']:
        tfrecord.main(args['<path>'])
