
import os, optparse

from r1_scrape_web import scrape_web
from r2_cluster_papers import cluster_papers
from r3_generate_html import generate_html

CONFERENCE = 'BMVC'
YEAR = 2013
BASE_FOLDER = os.path.expanduser("~/")+"Desktop/ConferenceGuides/"
N_TOPICS=5
N_TOP_WORDS=100

# if 0:

if __name__ == "__main__":

	parser = optparse.OptionParser()
	parser.add_option('-f', '--basefolder', dest='base_folder', default=BASE_FOLDER, help='base folder')
	parser.add_option('-c', '--conference', dest='conference', default=CONFERENCE, help='conference acronym')
	parser.add_option('-y', '--year', dest='year', default=YEAR, help='year')
	parser.add_option('-t', '--topics', dest='topics', default=N_TOPICS, help='# of topics')
	parser.add_option('-w', '--words', dest='words', default=N_TOP_WORDS, help='# of top words')
	(opt, args) = parser.parse_args()

	scrape_web(opt.base_folder, opt.conference, opt.year)
	cluster_papers(opt.base_folder, opt.conference, opt.year, opt.topics, opt.words)
	generate_html(opt.base_folder, opt.conference, opt.year)


