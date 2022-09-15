

def findnth(haystack, needle, n):
    c =1
    for i in range(len(haystack)):
        if c==n:
            return i
        if haystack[i] == needle:
            c += 1
    return -1


def break_line(inp, ch, n):
    if inp.count(ch)>n:
        break_index = findnth(inp, ch,n)
        return [inp[0:break_index]] + break_line(inp[break_index:], ch, n)
    else:
        return [inp]


lines = []

f=open("results.tsv","r")
line = f.readline()
while line:
    new_lines=break_line(line, '\t', 6)
    new_lines = [i.strip() for i in new_lines]
    new_lines = [i + "\tNA\tNA" if findnth(i, '\t', 6) == -1 else i for i in new_lines]
    lines += new_lines
    line=f.readline()
for line in lines:
    print(line)
