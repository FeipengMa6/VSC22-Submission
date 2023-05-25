
import jieba

CHINESE_ZHUCI = ['的', '了', '着', '吧', '啊', '呢']


def _partial_jaccard(s_q, s_t):
    s_intsec = s_t.intersection(s_q)

    return len(s_intsec) / len(s_q)


def _find_longest_commonstr(X, Y):
    m = len(X)
    n = len(Y)
    # Create a table to store lengths of
    # longest common suffixes of substrings.
    # Note that LCSuff[i][j] contains length
    # of longest common suffix of X[0..i-1] and
    # Y[0..j-1]. The first row and first
    # column entries have no logical meaning,
    # they are used only for simplicity of program
    LCSuff = [[0 for i in range(n + 1)]
              for j in range(m + 1)]

    # To store length of the
    # longest common substring
    length = 0

    # To store the index of the cell
    # which contains the maximum value.
    # This cell's index helps in building
    # up the longest common substring
    # from right to left.
    row, col = 0, 0

    # Following steps build LCSuff[m+1][n+1]
    # in bottom up fashion.
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                LCSuff[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                LCSuff[i][j] = LCSuff[i - 1][j - 1] + 1
                if length < LCSuff[i][j]:
                    length = LCSuff[i][j]
                    row = i
                    col = j
            else:
                LCSuff[i][j] = 0

    # if true, then no common substring exists
    if length == 0:
        resultStr = ''
    else:
        # allocate space for the longest
        # common substring
        resultStr = ['0'] * length

        # traverse up diagonally form the
        # (row, col) cell until LCSuff[row][col] != 0
        while LCSuff[row][col] != 0:
            length -= 1
            resultStr[length] = X[row - 1]  # or Y[col-1]

            # move diagonally up to previous cell
            row -= 1
            col -= 1

        # required longest common substring
        resultStr = ''.join(resultStr)

    return resultStr, length


def _find_longest_commonword(X, Y):
    intSec = Y.intersection(X)
    resultStr = ''
    maxLen = 0
    for word in intSec:
        if len(word) > maxLen:
            maxLen = len(word)
            resultStr = word

    return resultStr, maxLen


def _remove_longest_commonstr(X, Y):
    xWordSeg = list(jieba.cut(X))
    yWordSeg = list(jieba.cut(Y))
    xWordSegSet = set()
    yWordSegSet = set()
    for x in xWordSeg:
        x = ''.join(e for e in x if e.isalnum() and e not in CHINESE_ZHUCI)
        if x:
            xWordSegSet.add(x)
    for y in yWordSeg:
        y = ''.join(e for e in y if e.isalnum() and e not in CHINESE_ZHUCI)
        if y:
            xWordSegSet.add(y)
    xWordSegRst = set()
    yWordSegRst = set()

    resultStr, _ = _find_longest_commonstr(X, Y)
    resultWord, _ = _find_longest_commonword(xWordSegSet, yWordSegSet)
    resultStr = resultStr if len(resultStr) > len(resultWord) else resultWord
    for x in xWordSeg:
        if x not in resultStr:
            xWordSegRst.add(x)
    for y in yWordSeg:
        if y not in resultStr:
            yWordSegRst.add(y)

    return xWordSegRst, yWordSegRst


def _match_with_partial_jaccard(sq, st):

    sqWordSegRst, stWordSegRst = _remove_longest_commonstr(sq, st)
    if len(sqWordSegRst) == 0:
        return True
    elif _partial_jaccard(sqWordSegRst, stWordSegRst) > 0:
        return True
    else:
        return False

def rectify_query(query, title, kg):
    sim_query = [(query, 1.0)]
    if kg:
        try:
            sim_query.extend(kg.similar_by_word(query, topn=100))
            sim_query = [q for q in sim_query if q[1] >= 0.75]
        except Exception as e:
            sim_query = [(query, 1.0)]

    for item in sim_query:
        q, sim = item[0], item[1]
        if len(q) <= 3:
            if q in title:
                return q
        else:
            if _match_with_partial_jaccard(q, title):
                return q

    return None
