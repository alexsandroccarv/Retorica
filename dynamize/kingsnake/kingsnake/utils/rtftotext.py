# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import pyth.document
from pyth.plugins.plaintext.writer import PlaintextWriter
from kingsnake.utils.rtfreader import CustomRtf15Reader as Rtf15Reader


def can_be_converted_to_text(p):
    """Should return `True` if the given `pyth.document.Paragraph` can be
    converted to text by the PlaintextWriter, `False` otherwise.

    ...But currently only returns `False` for images.
    """
    return not isinstance(p, pyth.document.Image)


def sanitize_rtf_document(doc):
    """Sanitize a `pyth.document.Document`, removing everything that can't be
    converted to plain text.

    WARNING! This method operates in place, changing the input *doc* and
    returning `None`.
    """
    for paragraph in doc.content:
        paragraph.content = filter(can_be_converted_to_text, paragraph.content)


def extract_text_from_rtf_document(rtf):
    """Extract text from RTF document. *rtf* must be an open file-like RTF
    document, with its cursos placed at the beginning of the RTF data.
    """
    doc = Rtf15Reader.read(rtf)

    # Remove non textual elements from the RTF document
    sanitize_rtf_document(doc)

    # Convert the RTF document to plain text
    return PlaintextWriter.write(doc).read().decode('utf-8')
