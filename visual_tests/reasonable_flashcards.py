import random


# Function to read the file and split into terms and definitions
def read_and_split_file(file_path):
    terms = []
    definitions = []
    with open(file_path, 'r', encoding='utf-8') as file:  # Specify encoding as UTF-8
        for line in file:
            if '-' in line:
                term, definition = line.split('-', 1)  # Split on the first "-"
                terms.append(term.strip())
                definitions.append(definition.strip())
    return terms, definitions


# Function to simulate flashcard style interaction
def flashcard_interaction(terms, definitions):
    while True:
        index = random.randint(0, len(terms) - 1)  # Random index
        # Randomly choose whether to display the term or the definition
        if random.choice([True, False]):
            print(f"Term: {terms[index]}")
            user_input = input("Enter the matching definition: ")
            if user_input.lower() == definitions[index].lower():
                print("Correct!")
            else:
                print(f"The correct definition is: {definitions[index]}")
        else:
            print(f"Definition: {definitions[index]}")
            user_input = input("Enter the matching term: ")
            if user_input.lower() == terms[index].lower():
                print("Correct!")
            else:
                print(f"Incorrect! The correct term is: {terms[index]}")

        # Ask if the user wants to continue
        #continue_quiz = input("Continue? (y/n): ")
        #if continue_quiz.lower() != 'y':
        #    break


# Replace 'your_file_path.txt' with the path to your actual text file
file_path = 'reasonable_terms.txt'
terms, definitions = read_and_split_file(file_path)
flashcard_interaction(terms, definitions)