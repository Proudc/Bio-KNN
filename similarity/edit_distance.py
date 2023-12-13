
def edit_distance(x, y, memo=None):

    if memo is None:
        memo = [[0 for _ in range(len(y)+1)] for _ in range(len(x)+1)]

    # initialize first row and column
    for i in range(1, len(y)+1):
        memo[0][i] = i
    for i in range(1, len(x)+1):
        memo[i][0] = i

    # fill rest of matrix
    for i in range(1, len(x)+1):
        for j in range(1, len(y)+1):
            delta = 1 if x[i-1] != y[j-1] else 0
            memo[i][j] = min(
                memo[i-1][j-1] + delta,
                memo[i-1][j] + 1,
                memo[i][j-1] + 1
            )

    return memo[len(x)][len(y)], memo


def optimal_alignment(x, y, memo):

    i, j = len(x), len(y)
    result = ""  # prepend

    while i > 0 and j > 0:

        diag = memo[i-1][j-1]
        left = memo[i][j-1]
        abov = memo[i-1][j]

        # precedence: diagonal > left > above
        if diag <= left and diag <= abov:
            curr = 'M' if x[i-1] == y[j-1] else 'R'
        else:
            curr = 'I' if left <= abov else 'D'

        if curr in 'MRD':
            i -= 1
        if curr in 'MRI':
            j -= 1

        result = curr + result

    # edge case: haven't hit (0,0) yet
    result = 'D'*i + 'I'*j + result

    return result


def print_alignment(x, y, alignment):
    i, j = 0, 0
    for curr in alignment:

        if curr == 'I':
            x = x[:i] + '-' + x[i:]

        elif curr == 'D':
            y = y[:j] + '-' + y[j:]

        i, j = i+1, j+1

    return x, y


def cal_ed_between_seq(a, b):
    edit_dist, memo_map = edit_distance(a, b)
    operations          = optimal_alignment(a, b, memo_map)
    a, b                = print_alignment(a, b, operations)

    return edit_dist, a, b


if __name__ == '__main__':
    a = 'PTHIKWGD'
    b = 'KGDIMVFPR'
    edit_dist, a, b = cal_ed_between_seq(a, b)
    print(edit_dist)
    print(a)
    print(b)