# PAL
## History
### What is PAL?
PAL, or Peer-Assisted-Leadership, is an initiative of the University of Georgia Honors Program Student Council (HPSC). PAL provides an avenue for incoming UGA freshmen to connect with more experienced upperclassmen with similar professional and/or academic intentions. Interested students sign up by submitting a Google Form, providing information like their major(s), minor(s), and pre-professional track.

### What is this doing on GitHub?
Because a large number of students sign up for PAL, it would be difficult to pair off mentors and mentees by hand. Kyle Mercer, a former HPSC member, wrote the initial PAL sorting algorithm (PALgorithm) in JavaScript for [Node.js](https://nodejs.org/en/) using the [csvtojson](https://www.npmjs.com/package/csvtojson) module. Since its creation, HPSC members have continued to develop the algorithm and matching process; Monte Fischer rewrote the algorithm in Python and edited it significantly in an attempt to improve the strength of matched mentor/mentee pairs and make future algorithm tweaking easier, and Meredith Van De Velde continued to build upon the algorithm. Most recently, Elise Karinshak improved its matching capabilities by revising the degree category-assignment and similarity computation processes to account for multiple majors / minors. 

This algorithm will continue to be improved upon by further iterations of PAL.

## Details
### Running the PALgorithm
First, make sure you have both [Node.js](https://nodejs.org/en/) and the [csvtojson](https://www.npmjs.com/package/csvtojson) module installed. Then, download the Google Form data file, rename it data.csv, and place it in the same directory as pal.js and package.json. You may need to relabel the fields in data.csv depending on changes to the Google Form. Then run the terminal command `node pal.js > sorted.csv`. Open sorted.csv in your spreadsheet application of choice, and format to taste.
### Help! It doesn't work!
If you are struggling to get the PALgorithm to work, consult a technical-minded friend or get in touch with Elise (emk42835@uga.edu).
