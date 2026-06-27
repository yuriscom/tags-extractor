#!/usr/bin/python
import sys, getopt
from text_keyword_extractor import TagsExtractor
from htmlparser import MLStripper


def main(argv):
    inputfile = 'data/content3.txt'
    outputfile = 'data/test-output.json'
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print('cli.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('cli.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg or inputfile
        elif opt in ("-o", "--ofile"):
            outputfile = arg or outputfile

    f = open(inputfile, "r")
    text = f.read()
    f.close()

    s = MLStripper()
    s.feed(text)
    strippedText = s.get_data()

    tex = TagsExtractor()
    tex.extract(strippedText, num=20)
    tags = tex.to_json()
    print(tags)
    f = open(outputfile, "w")
    f.write(tags)
    f.close()


if __name__ == "__main__":
    main(sys.argv[1:])
