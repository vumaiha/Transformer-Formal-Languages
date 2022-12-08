import numpy as np

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

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

    def generate_bannedkgrams ():
        bannedkgrams = []
        bannedkgram_sigma = np.random.choice(self.sigma, self.nsigma_k, replace=False)
		while i < n_kgrams + 1:
			bannedkgram = ''
        	if self.nsigma_k == 1:  # type  is uniform
            	while len(bannedkgram)<self.k:
                	bannedkgram += bannedkgram_sigma[0]
				bannedkgrams.append(bannedkgram)
				i += 1
        	# elif:
            # 	if self.type == 'random':
            #     	while len(bannedkgram)<self.k:
            #         #write code to generate random string using all elements of bannedkgram_sigma
		print("List of banned k-grams:" + bannedkgrams)
        return bannedkgrams

    # def generate_string (self, bannedkgrams, length): #for now we will just generate strings where we directly input the banned kgram
    #     seq = get_sigma_star(self.sigma, length)  #generate a string in sigma*
    #     if self.belongs_to_lang (seq, bannedkgrams):
    #     	return seq

    def generate_list (self, num, min_length, max_length):
        length = np.random.randint(min_length, max_length)
        arr = []
        while len(arr) < num:
			string = self.get_sigma_star(self.sigma, length)
			if self.belongs_to_lang (string, self.bannedkgrams) && string not in arr:
				arr.append(string)
				print("Generated {}/{} samples".format(len(arr), num), end = '\r', flush = True)
		return arr

    def output_generator(self, seq):
        out_string = ''
		for i in range(1, len(seq)+1):
			part_seq = seq[:i]
			for sigma in self.sigma:
				if self.belongs_to_lang(part_seq + sigma, self.bannedkgrams):
					output_seq += '1'
				else:
					output_seq += '0'
        return out_string

    def training_set_generator (self, num, min_legnth, max_length):
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
