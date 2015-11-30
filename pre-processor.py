#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: ritesh
# @Date:   2015-11-29 17:47:44
# @Last Modified by:   ritesh
# @Last Modified time: 2015-11-29 20:06:13

import os
import re
import random
from email.parser import Parser


COLLECTION = "./spam-ham-dataset/processed-email-ham-spam-collection"
DATASET = "./spam-ham-dataset/split/"

def parse_payload(payload):
	if type(payload) == list:
		print payload
	mystring = payload.replace('\n', ' ').replace('\r', '')

	pattern =  r"\<.*?\>"
	replaced = re.sub(re.compile(pattern, re.MULTILINE), "url ", str(mystring))

	return replaced

	# # pattern =  r"(<)*http(s)*:\/\/.*(>)*?\s"
	# pattern =  r"\<.*\>?"
	# replaced = re.sub(re.compile(pattern, re.MULTILINE), "url ", str(payload))
	# # params = r"\s.*\>?$"
	# # replaced = re.sub(re.compile(params, re.MULTILINE), "params ", str(replaced))
	# mystring = replaced.replace('\n', ' ').replace('\r', '')
	# return mystring

def main():
	parser = Parser()

	ham_paths = [os.path.join("./spam-ham-dataset/split/", "HAM_%s.txt" %(i)) for i in xrange(1,101)]
	spam_paths = [os.path.join("./spam-ham-dataset/split/", "SPAM_%s.txt" %(i)) for i in xrange(1,11)]

	with open(COLLECTION, "w") as f:
		print ("""For hem paths, pre-processing""")
		for path in ham_paths:
			email = parser.parsestr(open(path).read())
			payload = email.get_payload()
			if email.is_multipart():
				while True:
					if type(payload) == str:
						break
					payload = payload[0].get_payload()

			# payload = email.get_payload()[0].get_payload() if email.is_multipart() else email.get_payload()
			"""parsing payload"""
			parsed_payload = parse_payload(payload)
			"""save in file with label"""
			f.write("%s\t%s\n" %("ham", parsed_payload))

		print ("""For spam paths, pre-processing""")
		for path in spam_paths:
			email = parser.parsestr(open(path).read())
			payload = email.get_payload()[0].get_payload() if email.is_multipart() else email.get_payload()
			"""parsing payload"""
			parsed_payload = parse_payload(payload)
			"""save in file with label"""
			f.write("%s\t%s\n" %("spam", parsed_payload))


if __name__ == '__main__':
	main()
