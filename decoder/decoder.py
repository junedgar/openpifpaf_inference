import logging
from abc import ABCMeta, abstractmethod, abstractstaticmethod


class Decoder(metaclass=ABCMeta):
    @abstractstaticmethod
    def match(head_names):  # pylint: disable=unused-argument
        return False

    @classmethod
    def cli(cls, parser):
        """Add decoder specific command line arguments to the parser."""

    @classmethod
    def apply_args(cls, args):
        """Read command line arguments args to set class properties."""

    @abstractmethod
    def __call__(self, fields):
        pass

def factory_decode(model, *, profile=None, **kwargs):
    """Instantiate a decoder for the given model.

    All subclasses of decoder.Decoder are checked for a match.
    """
    headnames = tuple(h.shortname for h in model.head_nets)

    for decoder in Decoder.__subclasses__():
        logging.debug('checking whether decoder %s matches %s',
                      decoder.__name__, headnames)
        if not decoder.match(headnames):
            continue
        logging.info('selected decoder: %s', decoder.__name__)
        return decoder(model.io_scales()[-1],
                       head_names=headnames,
                       **kwargs)

    raise Exception('unknown head nets {} for decoder'.format(headnames))
