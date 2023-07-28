# This function will swap characters for every permutation
def swap_char(word, i, j):
    swap_index = list(word)
    swap_index[i], swap_index[j] = swap_index[j], swap_index[i]

    return ''.join(swap_index)


# Driver code
def main():
    input_word = ["ab", "bad", "abcd"]
    indexes_to_swap = [(0, 1), (1, 2), (1, 3)]

    for i in range(len(input_word)):
        first_index, second_index = indexes_to_swap[i][0], indexes_to_swap[i][1]
        permuted_words = swap_char(input_word[i], first_index, second_index)

        print(i + 1, ".\t Input string: '", input_word[i], "'", sep="")
        print("\t Swapping character at index", first_index, "with index", second_index)
        print("\t Swapped indices: ",
              "[", ', '.join(permuted_words), "]", sep="")
        print('-' * 100)


if __name__ == '__main__':
    main()
