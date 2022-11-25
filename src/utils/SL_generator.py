import numpy as np

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def get_sigma_star(sigma, length):
	string = ''
	while len(string) < length:
		symbol = np.random.choice(sigma)
		string += symbol
	return string

class SLLanguage ():
    def __init__ (self, nsigma, k, nsigma_k, type):
        self.sigma = alphabet[:nsigma+1]
        self.char2id = {ch:i for i,ch in enumerate(self.sigma)}
        self.k = k
        self.nsigma_k = nsigma_k
        self.type = type #can be uniform, alternating, random

    def generate_bannedkgram ():
        bannedkgram = ''
        bannedkgram_sigma = np.random.choice(self.sigma, self.nsigma_k, replace=False)
        if self.nsigma_k == 1:  # type  is uniform
            while len(bannedkgram)<self.k:
                bannedkgram += bannedkgram_sigma[0]
        elif:
            if self.type == 'random':
                while len(bannedkgram)<self.k:
                    #write code to generate random string using all elements of bannedkgram_sigma
        return bannedkgram

    def generate_string (self, bannedkgram, length): #for now we will just generate strings where we directly input the banned kgram
        string = get_sigma_star(self.sigma, length)  #generate a string in sigma*
        while bannedkgram in string:
            string = get_sigma_star(self.sigma, length) #generat a new string in sigma*
        string =+ 'T' #concatenate terminal symbol onto string
        return string

    def generate_list (self, num, bannedkgram, max_length):
        length = np.random.randint(max_length + 1)
        arr = []
        while len(arr) < num:
			string = self.generate_string(bannedkgram, length)
			if string in arr:
				continue
			if len(string) <= max_length:
				arr.append(string)
				print("Generated {}/{} samples".format(len(arr), num), end = '\r', flush = True)
		return arr

    def output_generator(self, string): #need to understand what this supposed to generate, it is not simply hot encoding
        out_string = ''
        for ch in string:
            out_string += str(self.char2id[ch])

        return out_string

    def training_set_generator ():

    def lineToTensorOutput ():
