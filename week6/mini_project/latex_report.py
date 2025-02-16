from pylatex import (
    Document, 
    Section, 
    Subsection, 
    Command, 
    Figure, 
    Math, 
    Itemize, 
    PageStyle,
    simple_page_number,
    LineBreak,
    Head,
    Foot,
    Enumerate,
    LongTable,
    MultiColumn
)
from pylatex.utils import italic, NoEscape
import os

# add images
image_filename = os.path.join(os.path.dirname(__file__), "/home/scotty/dsc_207_week6/mini_project/plots/dis_of_num_data.png")
doc = Document()
def generate_header_footer(doc):
    header = PageStyle("header")
    # create header
    with header.create(Head("R")):
        header.append("Daffy Ducks")
        header.append(LineBreak())
        header.append("DSC 207: Mini-Project")
        header.append(LineBreak())
        header.append("Week 6 & 7")
    # add page number to footer in the center
    with header.create(Foot("C")):
        header.append(simple_page_number())
    # add header and footer to the document
    doc.preamble.append(header)
    doc.change_document_style("header")
generate_header_footer(doc)
with doc.create(Section('Mini-Project')):

    doc.append(italic('Goal: To explore the 2016.csv dataset and present findings using data analysis and visualization techniques.\
    In this mini-project, we will work with the dataset supplied and go through the data science process using the tools \
    and techniques learned from Weeks 1 - 5.The aim is to potentially use these tools together to achieve the objective of data exploration.'))

    with doc.create(Subsection('High Level View [2 pts]')):
        doc.append(italic('Describe the dataset in words (50 words). Look at the data samples and describe what they represent and how they could be useful in a variety of data science tasks.'))
        with doc.create(Enumerate()) as enum:
            enum.add_item('The 2016.csv contains 13 fields, and 157 records.The dataset gives us 157 unique countries, that are split across 10 different regions. We are given some metrics for each country like  happiness score, freedom, government corruption, family,and generosity. This data set could be useful for looking at how government corruption or freedom varies by region and country. It could also be useful for determining which of the metrics given is most influential on happiness score.')
with doc.create(Section('Data Exploration')):
    doc.append(italic('In this step, you should explore what is present in the data and how the data is organized.\
    You are expected to answer the following questions using the pandas library and markdown cells to describe your actions:'))               
    with doc.create(Subsection('Preliminary Exploration')):
        with doc.create(Enumerate()) as enum:
            enum.add_item(italic('Are there quality issues in the dataset (noisy, missing data, etc.)?'))
            with doc.create(Itemize()) as itemize:
                itemize.add_item('Checking for inital quality issues int the dataset with df.info() where df is a pandas DataFrame object of our 2016.csv. Printing df.info gives us the following information:')
                with doc.create(LongTable("l l l")) as data_table:
                    data_table.add_hline()
                    data_table.add_row(["column name", "total non-null values", "data type"])
                    data_table.add_hline() 
                    data_table.end_table_header()
                    data_table.add_hline()
                    data_table.add_row(["country", "157", "object"])
                    data_table.add_row(["region", "157", "object"])
                    data_table.add_row(["happiness_score", "157", "float64"])
                    data_table.add_row(["gdp_per_capita", "157", "float64"])
                    data_table.add_row(["family", "157", "float64"])
                    data_table.add_row(["life_expectancy", "157", "float64"])
                    data_table.add_row(["freedom", "157", "float64"])
                    data_table.add_row(["government_corruption", "157", "float64"])
                    data_table.add_row(["generosity", "157", "float64"])
                    data_table.add_row(["dystopia_residual", "157", "float64"])
                    data_table.add_hline()
                    data_table.add_row((MultiColumn(3, align="l", data="Table 1 : Initial quality check of the 2016.csv dataset"),))
                    data_table.end_table_last_footer() 
            enum.add_item(italic('What will you need to do to clean and/or transform the raw data for analysis?'))
            with doc.create(Itemize()) as itemize:
                itemize.add_item('Checking for null values (df.isnull().sum())')
                itemize.add_item('Dropping countries with a value 0 for an observation')
                itemize.add_item('The happiness_rank is an integer value that corresponds to the happiness_score, such that the max happiness is given a happiness_rank of 1, the min happiness is given a happiness_rank of n where n is the total number of records.Therefore, we need to reset the happiness rank to account for these dropped countries with an observation of 0.')
            enum.add_item(italic('What are trends in the dataset using descriptive statistics (mean, median etc) and distribution of numerical data (eg. histograms)?'))
            with doc.create(Itemize()) as itemize:
                itemize.add_item('In order to idenitify the global and statical trends in the data we can isolate the numerical columns categorical columns. By sepearting the numerical variables into a pandas dataframe we can \
                following descriptive statistics for each numerical variable: mean, median, and standard deviation. \
                We can compare mean and median to determine if the data is skewed.')
                itemize.add_item('By using histograms to visualize numerical data, we can see the distribution of the data. This will help us understand the data better and identify any skewness in these numerical variables.')
    with doc.create(Subsection('Preliminary Exploration Tasks')):
        doc.append(italic('You are expected to show a minimum of 2 preliminary exploration tasks that you performed with justification. Typically, preliminary exploration helps us in identifying specific objectives for data analysis tasks.'))
        with doc.create(Itemize()) as itemize:
            itemize.add_item('Check for skewness in the data using histograms and descriptive statistics')
            with doc.create(Enumerate()) as enum:
                enum.add_item('In a distribution that is skewed right, the mean is greater than the median. While in a distribution that is skewed left, the mean is less than the median. For distrobutions \
                that appear symmetric, the mean and median are roughly equal. By comparing the mean and median of the numerical variables in the 2016.csv dataset we can determine if a particular variable may contain outliers.')
                with doc.create(LongTable("l l l")) as data_table:
                    data_table.add_hline()
                    data_table.add_row(["column name", "mean", "median"])
                    data_table.add_hline() 
                    data_table.end_table_header()
                    data_table.add_hline()
                    data_table.add_row(["happiness_score", "5.382185", "5.314"])
                    data_table.add_row(["gdp_per_capita", "0.953880", "0.982"])
                    data_table.add_row(["family", "0.793621", "0.810"])
                    data_table.add_row(["life_expectancy", "0.557619", "0.606"])
                    data_table.add_row(["freedom", "0.370994", "0.397"])
                    data_table.add_row(["government_corruption", "0.137624", "0.088"])
                    data_table.add_row(["generosity", "0.242635", "0.222"])
                    data_table.add_row(["dystopia_residual", "2.325807", "2.290"])
                    data_table.add_hline()
                    data_table.add_row((MultiColumn(3, align="l", data="Table 2 : Mean and modes for numerical variables in the 2016.csv dataset"),))
                    data_table.end_table_last_footer() 
                enum.add_item('Histograms')
                with doc.create(Figure(position='h!')) as fig:
                    fig.add_image('/plots/dis_of_num_data.png', width='300px')
                    fig.add_caption('Figure 1 : Histograms of numerical data in the 2016.csv dataset')
            itemize.add_item('Check for null values or missing data in the dataset')
'''
        with doc.create(Subsection('Preliminary Exploration Tasks')):
            doc.append(italic('You are expected to show a minimum of 2 preliminary exploration tasks that you performed with justification. Typically, preliminary exploration helps us in identifying specific objectives for data analysis tasks.'))
            with doc.create(Itemize()) as itemize:
                itemize.add_item('Checking for null values (df.isnull().sum())')
                itemize.add_item('Histogram of distribution of happiness scores')
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
'''

doc.generate_tex('mini_project_outline')  # Generates the .tex file
doc.generate_pdf('mini_project_outline') # Compiles the .tex file to PDF. You'll need a LaTeX distribution installed. """
