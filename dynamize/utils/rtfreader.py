"""Here we have some customizations for the RTF reader we use, which is
provided by the `pyth` package.

Specifically, we needed to somehow instruct the reader to *handle* tables,
but we did only the necessary bits to correctly identify words inside tables,
so this is it: replace rows with newlines, replace cells with spaces.
"""
from pyth.plugins.rtf15 import reader


class CustomGroup(reader.Group):
    def handle_row(self):
        self.content.append(u'\n')

    def handle_cell(self):
        self.content.append(u' | ')


class CustomRtf15Reader(reader.Rtf15Reader):
    @classmethod
    def read(cls, source, errors='strict', clean_paragraphs=True):
        """
        source: A list of P objects.
        """

        reader = cls(source, errors, clean_paragraphs)
        return reader.go()

    def go(self):
        self.source.seek(0)

        if self.source.read(5) != r"{\rtf":
            from pyth.errors import WrongFileType
            raise WrongFileType("Doesn't look like an RTF file")

        self.source.seek(0)

        self.charsetTable = None
        self.charset = 'cp1252'
        self.group = CustomGroup(self)
        self.stack = [self.group]
        self.parse()
        return self.build()

    def parse(self):
        while True:
            next = self.source.read(1)

            if not next:
                break

            if next in '\r\n':
                continue
            if next == '{':
                subGroup = CustomGroup(self, self.group, self.charsetTable)
                self.stack.append(subGroup)
                subGroup.skip = self.group.skip
                self.group = subGroup
            elif next == '}':
                subGroup = self.stack.pop()
                self.group = self.stack[-1]
                subGroup.finalize()

                if subGroup.specialMeaning == 'FONT_TABLE':
                    self.charsetTable = subGroup.charsetTable
                self.group.content.append(subGroup)

            elif self.group.skip:
                # Avoid crashing on stuff we can't handle
                # inside groups we don't care about anyway
                continue

            elif next == '\\':
                control, digits = self.getControl()
                self.group.handle(control, digits)
            else:
                self.group.char(next)
