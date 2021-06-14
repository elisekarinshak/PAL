"""
Created on Sun Feb 10 17:05:28 2019
@author: DELLXPS
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


#VARIABLES

#Categorizing degrees:
# Major bins
businessBin = ["Accounting", "Accounting / International Business", "Actuarial Science Certificate", "Business", "Business & Political German Certificate", "Economics (A.B.)", "Economics (B.B.A.)", "Economics / International Business", "Entrepreneurship Certificate", "Environmental Economics & Management", "Finance", "Finance / International Business", "Financial Planning", "General Business (Griffin)", "General Business (Online)", "International Business", "Legal Studies Certificate", "Management", "Management / International Business", "Management Info Systems / Int'l Business", "Management Information Systems", "Marketing", "Marketing / International Business", "Pre-Business", "Pre-Law", "Real Estate", "Real Estate / International Business", "Risk Management & Insurance", "Risk Mgmt & Insurance / Int'l Business"]
spiaBin = ["Criminal Justice", "International Affairs", "Political Science", "Public Policy and Management"]
gradyBin = ["Advertising", "Communication Sciences & Disorders", "Communication Studies", "Entertainment and Media Studies", "Journalism", "New Media Certificate", "Pre-Journalism", "Public Relations", "Journalism - Visual Journalism"]
agricultureBin = ["Agribusiness", "Agribusiness Law Certificate", "Agricultural & Applied Economics", "Agricultural Communication", "Agricultural Education", "Agricultural Engineering", "Agriscience & Environmental Systems", "Agrosecurity Certificate", "Animal Health", "Animal Science", "Avian Biology", "Dairy Science", "Food Industry Marketing & Administration", "Food Science", "Horticulture", "Integrated Pest Management Certificate", "International Agriculture Certificate", "Local Food Systems Certificate", "Organic Agriculture Certificate", "Poultry Science"]
artBin = ["Art - Ceramics", "Art - Drawing", "Art - Fabric Design", "Art - Graphic Design", "Art - Jewelry & Metalwork", "Art - Painting", "Art - Photography", "Art - Printmaking", "Art - Scientific Illustration", "Art - Sculpture", "Art Education", "Art History", "Art X: Expanded Forms", "Furnishings & Interiors", "Interior Design", "Music", "Music Business Certificate", "Music Composition", "Music Performance", "Music Theory", "Music Therapy", "Studio Art", "Theatre", "Landscape Architecture", "Dance (A.B.)", "Dance (B.F.A.)", "Mass Media Arts"]
biologyBin = ["Biochemistry & Molecular Biology", "Biological Science", "Biology", "Cellular Biology", "Chemistry (B.S.)","Chemistry (B.S.Chem.)", "Environmental Chemistry", "Genetics", "Global Health Certificate", "Applied Biotechnology", "Microbiology", "Pharmaceutical Sciences", "Pharmacy", "Physics & Astronomy", "Plant Biology", "Pre-Dentistry", "Pre-Medicine", "Pre-Optometry", "Pre-Pharmacy", "Pre-Veterinary Medicine (B.S.)", "Pre-Veterinary Medicine (B.S.A.)", "Pre-Veterinary Medicine (B.S.F.R.)"]
computerBin = ["Cognitive Science", "Computer Science", "Computing Certificate", "Data Science", "Environmental Engineering", "Mathematics","Statistics", "Physics"]
humanitiesBin = ["Comparative Literature", "Film Studies", "History", "Honors Interdisc. Studies (A.B.)", "Honors Interdisc. Studies (B.S.)", "Honors Interdisc. Studies (B.S.A.)", "Honors Interdisc. Studies (B.S.F.C.S.)", "Interdisciplinary Studies (A.B.)", "Interdisciplinary Studies (A.B.)", "Interdisciplinary Studies (B.F.A.)", "Interdisciplinary Studies (B.S.)", "Interdisciplinary Writing Certificate","Linguistics", "Medieval Studies Certificate", "Native American Studies Certificate", "African American Studies", "African American Studies Certificate", "African Studies Certificate", "Anthropology", "Archaeological Sciences Certificate",  "Philosophy", "Pre-Theology", "Religion", "Women's Studies"]
cultureBin = ["Classical Culture", "Classical Languages","English","French", "German", "Germanic & Slavic Languages", "Greek","Chinese Language & Literature", "British & Irish Studies Certificate", "Italian", "Japanese Language & Literatures", "Latin American & Caribbean Studies", "Latin American & Caribbean Studies Cert.", "Arabic","Asian Studies Certificate","Romance Languages", "Russian", "Spanish", "Chinese Language and Literature"]
peopleBin = ["Dietetics", "Disability Studies Certificate", "Consumer Economics", "Consumer Foods", "Consumer Journalism", "Environmental Health Science", "Environmental Resource Science", "Exercise & Sport Science", "Family & Consumer Sciences Education", "Fashion Merchandising", "Health Promotion", "Housing Management and Policy", "Human Development and Family Science", "Leadership & Service Certificate", "Athletic Training", "Nutritional Sciences", "Personal & Org. Leadership Cert.", "Social Work", "Sociology", "Sport Management", "Turfgrass Management", "Psychology"]
educationBin = ["Early Childhood Education", "Educ. Psych & Instructional Tech Certif.", "English Education", "Health and Physical Education", "Mathematics / Mathematics Education", "Mathematics Education", "Middle School Education", "Music Education", "Science Education", "Special Education", "World Language Education", "Biology / Science Education", "English / English Education","History / Social Studies Education"]
natureBin = ["Ecology", "Ecology (A.B.)", "Ecology (B.S.)", "Fisheries & Wildlife", "Entomology", "Forestry", "Geographic Information Science Cert.", "Geography (A.B.)", "Geography (B.S.)", "Geology (A.B.)", "Geology (B.S.)", "Global Studies Certificate", "Community Forestry Certificate", "Natural Resource Recreation & Tourism", "Pre-Forest Resources", "Water & Soil Resources (B.S.E.S.)", "Water & Soil Resources (B.S.F.R.)", "Water Resources Certificate", "Coastal & Oceanographic Eng. Cert.", "Atmospheric Sciences", "Environmental Ethics Certificate"]
engineeringBin = ["Electrical and Electronics Engineering", "Engineering Physics Certificate", "Engineering Science Certificate", "Computer Systems Engineering", "Computer Systems Engineering Certificate", "Civil Engineering", "Mechanical Engineering", "Biochemical Engineering", "Biological Engineering"]
undecidedBin = ["Undecided"]

#store category bin labels in central bin
binBin = []
listOfBins = [businessBin, spiaBin, gradyBin, agricultureBin, artBin, biologyBin, computerBin, humanitiesBin, \
              cultureBin, peopleBin, educationBin, natureBin, engineeringBin]
for i in listOfBins:
    binBin.append(i)

major_dict = {'business': businessBin, 'spia':spiaBin, 'grady': gradyBin, 'agriculture': agricultureBin, 'art': artBin, \
    'biology': biologyBin, 'computer': computerBin, 'humanities': humanitiesBin, 'culture': cultureBin, \
    'people': peopleBin, 'education': educationBin, 'nature': natureBin, 'engineering': engineeringBin, \
    'undecided': undecidedBin}


#METHODS    
    
#Assigns category corresponding with degree type. Functional for multiple degrees within a cell.
#row: instance of data - corresponds with survey response
#label: either majors or minors; specifies type of degree to sort into categories
#returns degree categories
def find_categories(row, label):
    row_categories = []
    
    if pd.isnull(row[label]):
        return 'none'

    #for each cell in the major column
    for major_cell in row[label].split(';'): #separates majors by cell
        
        #parse major cell into list of majors
        majors = major_cell.split(', ')
        
        #assign category for each major
        category_cell = ""
        for major in majors:
            #for each category, search list of corresponding majors for specified major
            for category in major_dict.keys():
                #formats based on whether multiple majors
                if major in major_dict[category]:
                    if (category_cell == ""):
                        category_cell = category
                    elif (category_cell != category):
                        category_cell = category_cell + ", " + category
                    break
        
        #store list of categories in category cell
        row_categories.append(category_cell)
        
    return ';'.join(row_categories)
#find categories
    
#Find data containing previous match for mentee
#mentee_row: mentee to search for in previous data
#returns an observation of previous match data
def find_prev(mentee_row):
    prev = prev_matches[prev_matches["first_name_mentee"] == mentee_row["first_name"]]
    if (len(prev) != 0):
        prev = prev[prev["last_name_mentee"] == mentee_row["last_name"]]
    
    return prev
#find_prev

#Determines fit of matches by computing similarity. Compares majors, minors, 
#categories, and professional track.
#mentee: all mentee data
#mentor: all mentor data
#returns value of similarity cost function
def compute_similarity(mentee, mentor):
    #algorithm weights
    major_type = 5
    same_major_bonus = 3
    minor_type = 2
    same_minor_bonus = 1
    professional = 8
    rematch_penalty = -30
    max_score = major_type + same_major_bonus + minor_type + same_minor_bonus + professional
    
    # step 1, compute major similarity
    # Assign points for each same major category
    score = 0
    #for each cell in mentee major categories, split the categories and compare to split mentor categories
    #major categories
    for mentee_maj_cat in set(mentee['major_category'].split(";")):
        mentee_cats = mentee_maj_cat.split(", ")
        #compare mentee categories to mentor's categories
        for mentor_maj_cat in set(mentor['major_category'].split(";")):
            mentor_cats = mentor_maj_cat.split(", ")
            for cat in mentee_cats:
                if cat in mentor_cats:
                    score = score + major_type
    #same major bonus
    for mentee_maj in set(mentee['majors'].split(";")):
        mentee_majs = mentee_maj.split(", ")
        #compare mentee majors to mentor majors
        for mentor_maj in set(mentor['majors'].split(";")):
            mentor_majs = mentor_maj.split(", ")
            for major in mentee_majs:
                if major in mentor_majs:
                    score = score + same_major_bonus

    # step 2, compute pre-professional similarity
    if not pd.isnull(mentee['professional']) and not pd.isnull(mentor['professional']):
        if mentee['professional'] == mentor['professional']:
            score += professional

    # step 3, compute minor similarity
    if not pd.isnull(mentee['minors']) and not pd.isnull(mentor['minors']):
        
        #for each cell in mentee minor categories, split the categories and compare to split mentor minor categories
        for mentee_min_cat in set(mentee['minor_category'].split(";")):
            mentee_cats = mentee_maj_cat.split(", ")
            #compare mentee categories to mentor's categories
            for mentor_min_cat in set(mentor['minor_category'].split(";")):
                mentor_cats = mentor_min_cat.split(", ")
                for cat in mentee_cats:
                    if cat in mentor_cats:
                        score = score + minor_type
        #same minor bonus
        for mentee_min in set(mentee['minors'].split(";")):
            mentee_mins = mentee_min.split(", ")
            #compare mentee minors to mentor minors
            for mentor_min in set(mentor['minors'].split(";")):
                mentor_mins = mentor_min.split(", ")
                for minor in mentee_mins:
                    if minor in mentor_mins:
                        score = score + same_minor_bonus
    
    #step 4, check if previously matched
    if mentee['prev_participant'] == "Yes":
        prev_match = find_prev(mentee)
        if len(prev_match) != 0:
            if (mentor['first_name'] == prev_match['first_name_mentor'].iloc[0]) & (mentor['last_name'] == prev_match['last_name_mentor'].iloc[0]):               
                #mentee and mentor were matched in the past
                score = score + rematch_penalty   

    # this is, strictly speaking, a cost function
    return max_score - score
#compute_similarity


#CODE
    
#read in responses
#labels for the Google form survey data 
measured_attributes = ['timestamp', 'email', 'first_name', 'last_name', 'majors', 'minors', 'professional', 'phone', 'year', 'type', 'prev_participant', 'double_dawgs', 'fun_question']
form_responses = pd.read_csv('~/Desktop/Involvement/HPSC/PAL/sample_update.csv', delimiter=',', header=0, names=measured_attributes)
form_responses.sort_values(by='timestamp', inplace=True)

# drop all but the latest response of an individual
form_responses.drop_duplicates(['first_name', 'last_name'], keep='last', inplace=True)
# engineer additional features
form_responses['major_category'] = form_responses.apply(lambda row: find_categories(row, 'majors'), axis=1)
form_responses['minor_category'] = form_responses.apply(lambda row: find_categories(row, 'minors'), axis=1)
# ensure freshman are mentees and upperclassmen are mentors; reassign if necessary
form_responses.loc[(form_responses['year'] == 'Freshman'),'type']='Mentee'
form_responses.loc[(form_responses['year'] == 'Junior'),'type']='Mentor'
form_responses.loc[(form_responses['year'] == 'Senior'),'type']='Mentor'
# separate into mentors and mentees
mentees = form_responses.loc[form_responses['type'] == 'Mentee', :].reset_index(drop=True)
mentors = form_responses.loc[form_responses['type'] == 'Mentor', :].reset_index(drop=True)

#read in matches from past semester
measured_attributes = ["timestamp_mentee", "email_mentee", "first_name_mentee", "last_name_mentee", "majors_mentee	", "minors_mentee", "professional_mentee", "phone_mentee", "year_mentee", "type_mentee", "double_dawgs_mentee", "fun_question_mentee", "major_category_mentee", "minor_category_mentee",
                       "timestamp_mentor",	"email_mentor",	"first_name_mentor", "last_name_mentor", "majors_mentor	", "minors_mentor",	"professional_mentor", "phone_mentor",	"year_mentor", "type_mentor", "double_dawgs_mentor", "fun_question_mentor", "major_category_mentor", "minor_category_mentor"]
prev_matches = pd.read_csv('~/Desktop/Involvement/HPSC/PAL/pal_matches_sp21.csv', delimiter=',', header=0, names=measured_attributes)


#create match matrix
number_mentors = len(mentors)
number_mentees = len(mentees)
match_matrix = np.zeros([number_mentees, number_mentors])
for mentee_index, mentee_row in mentees.iterrows():
    for mentor_index, mentor_row in mentors.iterrows():
        # mentees are in the rows (we put mentees first!)
        match_matrix[mentee_index, mentor_index] = compute_similarity(mentee_row, mentor_row)

#determine matches from matrix that minimize cost
mentee_matches, mentor_matches = linear_sum_assignment(match_matrix)

#metrics to ensure successful matching
print("# mentors: ", number_mentors)
print("# mentees: ", number_mentees)
print("# mentee matches: ", len(mentee_matches))
print("# mentor matches: ", len(mentor_matches))
print("matches length equal: ", (len(mentee_matches) == len(mentor_matches)))

#store matched mentees and mentors
matched_mentees = mentees.iloc[mentee_matches].reset_index(drop=True)
matched_mentors = mentors.iloc[mentor_matches].reset_index(drop=True)
matches = matched_mentees.join(matched_mentors, lsuffix='_mentee', rsuffix='_mentor')
#Output file
matches.to_csv(path_or_buf='~/Desktop/matches.csv', index=False)

#unmatched mentees
ix = [i for i in mentees.index if i not in mentee_matches]
unmatched_mentees = mentees.loc[ix].reset_index(drop=True)
#Output file
unmatched_mentees.to_csv(path_or_buf='~/Desktop/unmatched_mentees.csv', index=False)
print('# unmatched mentees: ', len(unmatched_mentees))
