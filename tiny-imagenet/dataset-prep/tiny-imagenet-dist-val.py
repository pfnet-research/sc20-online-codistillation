import sys
import os


def main():
    basedir, destdir = sys.argv[1:]
    print('base val directory:', basedir, file=sys.stderr)
    print('destination directory:', destdir, file=sys.stderr)
    annotation_name = os.path.join(basedir, 'val_annotations.txt')

    clss = set()
    copys = []

    with open(annotation_name, 'r') as fp:
        for line in fp.readlines():
            img, cls = line.split('\t')[:2]
            clss.add(cls)
            copys.append((img, cls))

    print('# classes: {}'.format(len(clss)), file=sys.stderr)
    for cls in clss:
        print('mkdir {}'.format(os.path.join(destdir, cls)))
    for img, cls in copys:
        src = os.path.join(basedir, 'images', img)
        dest = os.path.join(destdir, cls, img)
        print('cp {} {}'.format(src, dest))


if __name__ == '__main__':
    main()
