'''
Script for scraping SIAM Data Mining papers from the web
'''

import urllib2
import json
from BeautifulSoup import BeautifulSoup as bs
import requests
import os
import optparse

base_folder = os.path.expanduser("~/")+"Dropbox/ConferenceGuides/"

def verifiy_folder_hierarchy(folder_name):
	folder_name = folder_name.split("/")
	for i in range(2,len(folder_name)):
		uri = "/".join(folder_name[:i])
		if not os.path.isdir(uri):
			os.mkdir(uri)
			print "Created folder:", uri


CONFERENCE = 'SDM'
YEAR = 2013

base_url = "http://knowledgecenter.siam.org/sdm-toc/"
year = YEAR
conference = CONFERENCE

page = urllib2.urlopen(base_url)
soup = bs(page.read())

# Extract titles, authors, and pdf URLs
blocks = soup.findAll("li")
papers = []
for i in range(len(blocks)):
	# x = blocks[18]
	# i=18
	try:
		tmp = blocks[i].find("span", "t-field t-new t-index-main").find("a")
		url = tmp['href'] + "/~~PdfSource/0"
		title = tmp.text
		authors_raw = blocks[i].findAll("span", "t-listitem t-listitem-allauthors")
		authors = [x.text for x in authors_raw]

		if len(authors) > 0:
			papers += [{"Title":title, "Authors":authors, "PDF":url,
						"Conference":"SDM", "Year":2013}]
	except:
		pass

pdf_folder = base_folder + "{}/{}/pdfs/".format(year, conference)

# Make sure all folders have been created
verifiy_folder_hierarchy(pdf_folder)

# Load config file or make a new one
# all_papers = cPickle.load(open("BMVC_paper_data.pkl", 'r'))
filename = base_folder + "paper_names/{}{}_papers.json".format(conference, year) 
json.dump(papers, open(filename, 'w'), indent=4)


# Download papers
for i in range(len(papers)):
	# Only check if we don't have a URL on file
	if papers[i]['PDF'] != '':
		title = papers[i]['Title'].encode("ascii", "ignore")
		authors = map(lambda x:x.encode("ascii", "ignore"), papers[i]['Authors'])
		authors = ",".join(authors)
		url = papers[i]['PDF']
		
		try:
			req = requests.get(url)
		except:
			print "Couldn't download url:", url
			continue

		# Save the paper in the appropriate folder
		if len(req.content) > 15:
			name = title + " - " + authors + ".pdf"
			with open(pdf_folder+name, "wb") as f:
				f.write(req.content)

		print "#{} of {}".format(i, len(papers))

