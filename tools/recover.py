def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--old-anno-file',required=None)
    parser.add_argument('--new-anno-file',default=None)
    parser.add_argument('--lmkfile',default=None)
    parser.add_argument('--boxfile', default=None)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    if args.old_anno_file is None or args.new_anno_file is None:
        exit(0)
    lines = open(args.old_anno_file,'r').readlines()
    if args.lmkfile is None:
        lmkLines = None
    else:
        lmkLines = open(args.lmkfile,'r').readlines()
    if args.boxfile is None:
        boxLines = None
    else:
        boxLines = open(args.boxfile,'r').readlines()



    if lmkLines is not None:
        assert len(lmkLines) == len(lines)
        for i, line, lmk in enumerate(lines, lmkLines):
            line = line.strip().split()
            lmk = lmk.strip().split()
            assert line[0] == lmk[0]
            line[1:106*2+1] = lmk[1:]
            lines[i] = " ".join(line) +'\n'

    if boxLines is not None:
        assert len(boxLines) == len(lines)
        for i, line, box in enumerate(lines, boxLines):
            line = line.strip().split()
            box = box.strip().split()
            assert line[0] == box[0]
            line[106*2+1+4:1:106*2+1+4+4] = box[1:]
            lines[i] = " ".join(line)+'\n'

    outf = open(args.new_anno_file, 'w')
    for line in lines:
        outf.write(line)
    outf.close()



