
import os
from xml.sax import make_parser, ContentHandler


class ParserXML(ContentHandler):
    __instance = None

    @classmethod
    def instance(cls):
        if not cls.__instance:
            obj = cls()
            cls.__instance = obj
        return cls.__instance

    # def __new__(cls, *args, **kwargs):
    #     if not cls.__instance:
    #         obj = super(ParserXML, cls).__new__()
    #         cls.__instance = obj
    #     return cls.__instance

    def __init__(self):
        super(ParserXML, self).__init__()
        self.data_dic = {}
        self.filename = ''
        self.vector = []
        self.size = []

    def _load_data(self, path):
        self.size = []
        parser = make_parser()
        parser.setContentHandler(self)
        parser.parse(path)

    def startElement(self, name, attrs):
        self.name  = name

    def characters(self, content):
        if '\t' not in content and '\n' not in content:
            if self.name == 'filename':
                self.data_dic[content] = []
                self.filename = content
            if self.name in ['width', 'height']:
                self.size.append(int(content))
            if content == 'pingpang':
                self.vector = [1, 1, 0]
                self.vector.extend(self.size)
            if content == 'xiangji':
                self.vector = [1, 0, 1]
                self.vector.extend(self.size)
            if self.name in ['xmin', 'ymin', 'xmax', 'ymax']:
                self.vector.append(int(content))

    def endElement(self, name):
        if name == 'ymax':
            self.data_dic[self.filename].append(self.vector)


if __name__ == '__main__':
    xml_folder = r'G:\Data\yoloV4自己的数据集\Annotations'
    parser = ParserXML()
    for xml in os.listdir(xml_folder):
        xml_file = os.path.join(xml_folder, xml)
        parser._load_data(xml_file)
    print(parser.data_dic)


