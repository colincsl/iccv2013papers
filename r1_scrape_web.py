'''
Script for scraping BMVC papers from the web

As of September 11, 2013 only 37 of 131 (28%) papers are available

Usage: simply run this file from the command line
'''

import os
from GoogleScraper import scrape
import requests
import json
import optparse

CONFERENCE = 'ICCV'
YEAR = 2011
BASE_FOLDER = os.path.expanduser("~/")+"Dropbox/ConferenceGuides/"


def verifiy_folder_hierarchy(folder_name):
	folder_name = folder_name.split("/")
	for i in range(2,len(folder_name)):
		uri = "/".join(folder_name[:i])
		if not os.path.isdir(uri):
			os.mkdir(uri)
			print "Created folder:", uri

def scrape_web(base_folder, conference, year):
	pdf_folder = base_folder + "{}/{}/pdfs/".format(year, conference)

	# Make sure all folders have been created
	verifiy_folder_hierarchy(pdf_folder)

	# Load config file or make a new one
	# all_papers = cPickle.load(open("BMVC_paper_data.pkl", 'r'))
	filename = base_folder + "paper_names/{}{}_papers.json".format(conference, year) 
	all_papers = json.load(open(filename))
	print 
	print "JSON data loaded from", filename
	print 

	# Find out if any of the papers are stored online
	print "Checking if PDFs of the papers can be found online"
	print 
	if 1:
		for i in range(len(all_papers)):
			# Only check if we don't have a URL on file
			if all_papers[i]['PDF'] == '':
				try:
					title = all_papers[i]['Title'].encode("ascii", "ignore")
					authors = all_papers[i]['Authors'].encode("ascii", "ignore")
					urls = scrape(""" "{}"" {} """.format(title, authors), number_pages=1, results_per_page=10)
					pdfs = [x.netloc+x.path for x in urls if x.path.find(".pdf")>0]
					if len(pdfs) > 0:
						pdf = pdfs[0]
						pdf = pdf[:pdf.find(".pdf")+4]
						all_papers[i]['PDF'] = pdf
						print "Found URL for: ", title
				except:
					json.dump(all_papers, open(filename, 'w'), indent=4)
					print "Error", i
					# exit()
					break

	# Save the data
	print "Saving the config file to ", filename
	print 
	# cPickle.dump(all_papers, open("BMVC_paper_data.pkl", 'w'))

	json.dump(all_papers, open(filename, 'w'), indent=4)

	from IPython import embed
	# Download each paper that we don't have
	print "Downloading the PDFs"
	print 
	paper_count = 0
	for i in range(len(all_papers)):
		if all_papers[i]['PDF'] != '':
			paper_count += 1
			name = all_papers[i]['Title'] + " - " + all_papers[i]['Authors'] + ".pdf"

			# Check if the paper has already been downloaded
			if os.path.exists(pdf_folder+name):
				continue
			
			# Get the paper from online
			print "Downloading #{}:".format(i), name
			# embed()
			url = all_papers[i]['PDF']
			# embed()
			if url.find("http") < 0:
				url = "http://"+url
			print url
			try:
				req = requests.get(url)
			except:
				print "Couldn't download url:", url
				continue

			# Save the paper in the appropriate folder
			if len(req.content) > 15:
				with open(pdf_folder+name, "wb") as f:
					f.write(req.content)

			print "#{} of {}".format(i, len(all_papers))

	print "{} of {} papers are downloaded".format(paper_count, len(all_papers))



if __name__ == "__main__":

	parser = optparse.OptionParser()
	parser.add_option('-f', '--basefolder', dest='base_folder', default=BASE_FOLDER, help='')
	parser.add_option('-c', '--conference', dest='conference', default=CONFERENCE, help='')
	parser.add_option('-y', '--year', dest='year', default=YEAR, help='')
	(opt, args) = parser.parse_args()


	scrape_web(opt.base_folder, opt.conference, opt.year)

