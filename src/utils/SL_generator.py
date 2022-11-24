import numpy as np

sigma = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

class SLLanguage ():
    def __init__ (self, nsigma, k, nsigma_k, type):
        self.sigma = sigma[:nsigma+1]
        self.k = k
        self.nsigma_k = nsigma_k
        self.type = type %can be uniform, alternating, random

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

    def generate_string (bannedkgram): #for now we will just generate strings where we directly input the banned kgram
        string = ''

        return string

    def generate_list ():

    def training_set_generator ():

    def lineToTensorOutput ():
