from utils import TextShuffler, pre_caption

class TextShufflerTester:
    
    @staticmethod
    def test_shuffler():
        example_text = "The lungs are clear there is no pleural effusion or pneumothorax"

        shuffler = TextShuffler()

        print("Original Text:")
        print(example_text)
        
        example_text = pre_caption(example_text,50)
        print("\nShuffled Nouns and Adjectives:")
        print(shuffler.shuffle_nouns_and_adj(example_text))
        print("\nShuffled All Words:")
        print(shuffler.shuffle_all_words(example_text))
        print("\nShuffled All but Nouns and Adjectives:")
        print(shuffler.shuffle_allbut_nouns_and_adj(example_text))
        print("\nShuffled Within Trigrams:")
        print(shuffler.shuffle_within_trigrams(example_text))
        print("\nShuffled Trigrams:")
        print(shuffler.shuffle_trigrams(example_text))
        print("\nReversed Sentence:")
        print(shuffler.reverse_sentence(example_text))
        print("\nShuffled Nouns, Verbs, and Adjectives:")
        print(shuffler.shuffle_nouns_verbs_adj(example_text))
        print("\nReplace Adjectives with Antonyms:")
        print(shuffler.replace_adjectives_with_antonyms(example_text))
        print("\nSwap adjacent words in the sentence:")
        print(shuffler.swap_adjacent_words(example_text))

# Run the test
TextShufflerTester.test_shuffler()
