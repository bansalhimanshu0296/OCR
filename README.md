# OCR

This project was done as a part of CSCI-B-551 Elements of Artificial Intelligence Coursework under Prof. Dr. David Crandall.

## Command to run the program ##

python3 ./image2text.py [training image file] [train text file] [test image file]

## Approach and design decisions

We have used a grid match function to match the grid of characters. For this we have iterated through each row of the grids and cols of grid matching every character in that if it’s a match, we have checked if it’s a start (‘*’) or space (‘ ‘) by increasing the variables separately.

For the compute emission function, we have calculated the total pixels. We have then iterated through each index of character in test letters and through each character in the training letter. Then we have called the grid match function on the test letters and train letter grid to calculate the space and star matches. Total unmatched is the difference between the total pixels and matched pixels. Then we have assigned different weights to different probabilities to reduce the noise. Going forward, we have multiplied all the probabilities for each pixel and adding it to the probabilities as they are calculated in log. Then we have set the emit probability of the character at index i.

Next for the train function, we have read all the lines. If the initial word is not present in the initial probability dictionary, it initializes it. We have then increased the count of the initial probability by 1. If the character is not present in the character probability dictionary, it initializes it. We have then increased the character count by 1. We have then looped through each character of the line and found the transition counts for each character and count of each character. Then we have calculated the total sum of characters and converted each character count to its probability. If the character was not present, the probability is set to -inf. Then we have calculated the total sum of initial characters and converted each character count to its probability. If the character was not present, the probability is set to -inf. At last, we have converted the number of transitions to transition probability for each character, if it’s not present, set it to -inf.

For the simplified method, we have iterated through each index in the image. We have set the best character as space and the best probability as minimum value. Then we have found the char and prob of that character and their emission probability dictionary for index i of sentence. We have then found the final probability, if the final probability is greater than best probability, then we have swapped it with the current probability and the current character. We have then appended this best char to the predicted output.

For HMM, we have initialized a V table and which table of size N which is the number of characters. We have then iterated through each character and setting its probability for 0 index in V table and 0 index in each character. We have then iterated through each index of character in image and through each independent character and found the which table and V table for each character and i index which is the most probable character from which we can come to this character with what probability. We have then multiplied the emission probability of that char at i index and added in case of log. We have then initialized an empty list of size N and the best probability as -inf. We have then iterated through each character on the final index finding the best suitable character for the last index based on the probability stored in V table. Then we have iterated through each index of image from last and found the best char from which through which we can reach the character. We have returned the characters by joining them.

**Assumption:** We have assigned different arbitrary weights to character match probability.



## Challenges

Finding the emission probability was quite difficult in this problem. 
