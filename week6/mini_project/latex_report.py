from pylatex import Document, Section, Subsection, Command, Figure, Math, Itemize

doc = Document()

with doc.create(Section('Mini-Project - Week 6 & 7')):

    doc.append('Note: This project is due at the end of week 7. There is no weekly assignment for week 6 but there will be a weekly assignment for week 7.')

    doc.append('In this mini-project, you will work with a dataset and go through the data science process using the tools and techniques learned from Weeks 1 - 5. The aim is to potentially use these tools together to achieve the objective of data exploration. The samples are for your reference but you may choose to do things differently.')

    with doc.create(Subsection('High Level View [2 pts]')):
        doc.append('Describe the dataset in words (50 words). Look at the data samples and describe what they represent and how they could be useful in a variety of data science tasks.')

    with doc.create(Subsection('Preliminary Exploration [4 pts]')):
        doc.append('In this step, you should explore what is present in the data and how the data is organized.')
        doc.append('You are expected to answer the following questions using the pandas library and markdown cells to describe your actions:')
        with doc.create(Itemize()) as itemize:
            itemize.add_item('Are there quality issues in the dataset (noisy, missing data, etc.)?')
            itemize.add_item('What will you need to do to clean and/or transform the raw data for analysis?')
            itemize.add_item('What are trends in the dataset using descriptive statistics (mean, median etc) and distribution of numerical data (eg. histograms)?')

        doc.append('You are expected to show a minimum of 2 preliminary exploration tasks that you performed with justification. Typically, preliminary exploration helps us in identifying specific objectives for data analysis tasks (Step 3).')
        doc.append('Sample:')
        with doc.create(Itemize()) as itemize:
            itemize.add_item('Checking for null values (df.isnull().sum())')
            itemize.add_item('Histogram of distribution of happiness scores')


    with doc.create(Subsection('Defining objectives - [3 pts]')):
        doc.append('Now that you have a better understanding of the data, you will want to form a research question which is interesting to you. The research question should be broad enough to be of interest to a reader but narrow enough that the question can be answered with the data. Some examples:')
        with doc.create(Itemize()) as itemize:
            itemize.add_item('Too Narrow: What is the GDP of the U.S. for 2011? This is just asking for a fact or a single data point.')
            itemize.add_item('Too Broad: What is the primary reason for global poverty? This could be a Ph.D. thesis and would still be way too broad. What data will you use to answer this question? Even if a single dataset offered an answer, would it be defendable given the variety of datasets out there?')
            itemize.add_item('Good: Can you use movie duration in a movie database to analyze viewer behavior over the years? If you have, or can obtain, data on a variety of movies and you have their box office earnings, this is a question that you can potentially answer well.')

        doc.append('You are expected to define a minimum of 3 objectives for the mini-project.')
        doc.append('Sample objectives -')
        with doc.create(Itemize()) as itemize:
            itemize.add_item('How closely are GDP per capita and Happiness scores related? Or slightly more general, how are different factors correlated in determining happiness scores.')
            itemize.add_item('What are some of the geographies that have higher/lower happiness scores? Can they be aggregated or categorized by region?')


    with doc.create(Subsection('Present Your Findings [9 pts]')):
        doc.append('This step involves using libraries like numpy and pandas to extract data from the main dataset to forms that help answer the objectives - common applications would be filtering, aggregation, data modification, augmentation etc. The data analysis should allow you to create visualizations that make the report informative and easy to read.')

        doc.append('You are required to present a minimum of 3 data analysis tasks and accompanying visualizations (one for each question) but any supporting visualizations can also be added. Visualizations should include -')
        with doc.create(Itemize()) as itemize:
            itemize.add_item('Justification for choice of plot')
            itemize.add_item('Plot (with appropriate details and aesthetics)')
            itemize.add_item('Inference/Conclusion from the visualization')

        doc.append('The data analysis and visualization may not be done strictly in order. You may choose to report findings in a way that is easy to understand and read.')
        doc.append('Sample visualizations -')
        with doc.create(Itemize()) as itemize:
            itemize.add_item('Heatmaps & Scatterplots for correlation trends')
            itemize.add_item('Bar graphs/Pie Charts for categorical data')

    with doc.create(Subsection('Ethics [2 pts]')):
        doc.append('Describe in words, or supporting visualization minimum 1 ethical concern you observe in the dataset.')
        doc.append('Sample : Can we be confident that some of the features like ‘freedom’ can be represented numerically? Can there be bias?')


doc.generate_tex('mini_project_outline')  # Generates the .tex file
doc.compile() # Compiles the .tex file to PDF. You'll need a LaTeX distribution installed.