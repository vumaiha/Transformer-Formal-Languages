import numpy as np

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
seed = np.random.randint (0,100000000)

def get_sigma_star(sigma, length):
    string = ''
    while len(string) < length:
        symbol = np.random.choice(sigma)
        string += symbol
    return string

class SLLanguage ():
    def __init__ (self, n_letters, k, nsigma_k, n_kgrams, type):
        self.n_letters = n_letters
        self.sigma = alphabet[:n_letters] #alphabet of the size of n_letters
        self.char2id = {ch:i for i,ch in enumerate(self.sigma)} #dictionary assigning each character to a id number
        self.k = k #length of banned k-grams
        self.nsigma_k = nsigma_k #size of the alphabet in the k-grams
        self.n_kgrams = n_kgrams #number of k-grams
        self.type = type #can be uniform, alternating, random
        self.bannedkgrams = self.generate_bannedkgrams ()

    def belongs_to_lang (self, seq, bannedkgrams):
        for bannedkgram in bannedkgrams:
            if bannedkgram in seq:
                return False
        return True

    def generate_bannedkgrams (self):
        bannedkgrams = []
        np.random.seed(seed)
        bannedkgram_sigma = np.random.choice(self.sigma, self.nsigma_k, replace=False)
        i = 0
        while i < self.n_kgrams:
            bannedkgram = ''
            if self.nsigma_k == 1:  # type  is uniform
                while len(bannedkgram)<self.k:
                    bannedkgram += bannedkgram_sigma[0]
                bannedkgrams.append(bannedkgram)
            elif self.type == 'alternating' and self.k%len(bannedkgram_sigma)==0:
                kgramunit = ''
                print(' '.join(bannedkgram_sigma))
                kgramunit = ''.join(bannedkgram_sigma)
                print(kgramunit)
                while len(bannedkgram)<self.k:
                    bannedkgram += kgramunit
                bannedkgrams.append(bannedkgram)
            elif self.type == 'random':
                bannedkgram = ''.join(bannedkgram_sigma)
                while len(bannedkgram)<self.k:
                    bannedkgram += np.random.choice(bannedkgram_sigma)
                bannedkgram_list = list(bannedkgram)
                np.random.shuffle(bannedkgram_list)
                bannedkgram = ''.join(bannedkgram_list)
                bannedkgrams.append(bannedkgram)
            i += 1
            # elif:
            #     if self.type == 'random':
            #         while len(bannedkgram)<self.k:
            #         #write code to generate random string using all elements of bannedkgram_sigma
        print("List of banned k-grams:" + ' '.join(bannedkgrams) + '\n')
        return bannedkgrams

    def generate_string (self, bannedkgrams, min_length, max_length):
        string = ''
        bannedkgram = bannedkgrams[0]  # assuming there is only one k-gram
        np.random.seed()
        poss_sigma = self.sigma.copy()
        poss_sigma.remove(bannedkgram[-1])
        length = np.random.randint(min_length, max_length)
        string += np.random.choice(self.sigma)
        while len(string) < length:
            if string[-(self.k-1):] == bannedkgram[:-1]:
                string += np.random.choice(poss_sigma)
            else:
                string += np.random.choice(self.sigma)
        return string

    def generate_list (self, num, min_length, max_length):
        arr = []
        i = 0
        while len(arr) < num and i < 1000000000: #stops generating strings after 1 million failures
            print("i={}".format(i), end='\r', flush=True)
            # np.random.seed()
            # length = np.random.randint(min_length, max_length)
            # print('Length: {}, min: {}, max {}'.format(length, min_length, max_length), end = '\r', flush = True)
            # string = get_sigma_star(self.sigma, length)
            string = self.generate_string (self.bannedkgrams, min_length, max_length)
            if self.belongs_to_lang (string, self.bannedkgrams) and string not in arr:
                arr.append(string)
                print("Generated {}/{} samples".format(len(arr), num), end = '\r', flush = True)
            else:
                i += 1
        return arr

    def output_generator(self, seq):
        output_seq = ''
        for i in range(1, len(seq)+1):
            part_seq = seq[:i]
            for sigma in self.sigma:
                if self.belongs_to_lang(part_seq + sigma, self.bannedkgrams):
                    output_seq += '1'
                else:
                    output_seq += '0'
        return output_seq

    def training_set_generator (self, num, min_length, max_length):
        input_arr = self.generate_list (num, min_length, max_length)
        output_arr = []
        for seq in input_arr:
            output_arr.append (self.output_generator (seq))
        return input_arr, output_arr

    def lineToTensorOutput(self, line):
        tensor = torch.zeros(len(line), self.n_letters)
        for li, letter in enumerate(line):
            letter_id = self.char2id[letter]
            tensor[li][letter_id] = 1.0
        return tensor

    def depth_counter(self, seq):
        ## To Do. The current implementation is not right, just a placeholder
        return np.ones((len(seq), 1))