Prediction of Song Decade based on Song's Audio Features

Name: Alex Link

Abstract

This analysis explores which models accurately predict the decade in
which a song was produced using audio features made up of both
categorical and numerical data.

Introduction

Jeff Titon, professor emeritus of music at Brown University, has
described ethnomusicology as the study of people making music; making
the sounds they call music and making music as a cultural domain. In
more simplified terms, ethnomusicology looks at music as a reflection of
culture. If music is a reflection of culture, can significant changes in
auditory features be used to accurately identify the changes in culture?
And if so, which features are most associated with different cultural
periods? The goal of this study was to create a model to correctly
classify music into the decade it was produced, using the audio features
that comprise each song, and see if this could shed some light on how
music has evolved over the past one hundred years.

Data

Our dataset was downloaded from Kaggle[^1], but the data itself was
originally pulled from Spotify via their Web API in December 2020. It
includes over 174,000 tracks, ranging from release years of 1920 to
2020, each containing a variety of metadata (such as artist name, track
name, Spotify track ID and release date), as well as various audio
features, described in detail in the table below.

  ------------------ ------------------------------------ ----------------- --------------
  Feature            Description                          Type              Range

  Acousticness       A confidence measure of whether the  Float             0.0 to 1.0
                     track is acoustic. 1.0 represents                      
                     high confidence the track is                           
                     acoustic.                                              

  Danceability       Danceability describes how suitable  Float             0.0 to 1.0
                     a track is for dancing based on a                      
                     combination of musical elements                        
                     including tempo, rhythm stability,                     
                     beat strength, and overall                             
                     regularity. A value of 0.0 is least                    
                     danceable and 1.0 is most danceable.                   

  Duration           Duration of the track in             Integer           4,937 to
                     milliseconds.                                          5,338,302

  Energy             Represents a perceptual measure of   Float             0.0 to 1.0
                     intensity and activity. Typically,                     
                     energetic tracks feel fast, loud,                      
                     and noisy. For example, death metal                    
                     has high energy, while a Bach                          
                     prelude scores low on the scale.                       
                     Perceptual features contributing to                    
                     this attribute include dynamic                         
                     range, perceived loudness, timbre,                     
                     onset rate, and general entropy.                       

  Explicit           Whether or not the track has         Integer/ Boolean  0 or 1
                     explicit lyrics (1=yes it does;0= no                   
                     it does not OR unknown)                                

  Instrumentalness   Predicts whether a track contains no Float             0.0 to 1.0
                     vocals. "Ooh" and "aah" sounds are                     
                     treated as instrumental in this                        
                     context. Rap or spoken word tracks                     
                     are clearly "vocal". The closer the                    
                     instrumentalness value is to 1.0,                      
                     the greater likelihood the track                       
                     contains no vocal content. Values                      
                     above 0.5 are intended to represent                    
                     instrumental tracks, but confidence                    
                     is higher as the value approaches                      
                     1.0.                                                   

  Key                The key the track is in. Integers    Integer           0 to 11
                     map to pitches using standard Pitch                    
                     Class notation . E.g. 0 = C, 1 =                       
                     C♯/D♭, 2 = D, and so on.                               

  Liveness           Detects the presence of an audience  Float             0.0 to 1.0
                     in the recording. Higher liveness                      
                     values represent an increased                          
                     probability that the track was                         
                     performed live. A value above 0.8                      
                     provides strong likelihood that the                    
                     track is live.                                         

  Loudness           The overall loudness of a track in   Float             -60.0 to 3.855
                     decibels (dB). Loudness values are                     
                     averaged across the entire track and                   
                     are useful for comparing relative                      
                     loudness of tracks. Loudness is the                    
                     quality of a sound that is the                         
                     primary psychological correlate of                     
                     physical strength (amplitude).                         
                     Values typical range between -60 and                   
                     0 db.                                                  

  Mode               Indicates the modality (major or     Integer/Boolean   0 or 1
                     minor) of a track, the type of scale                   
                     from which its melodic content is                      
                     derived. Major is represented by 1                     
                     and minor is 0.                                        

  Popularity         The popularity is calculated by      Integer           0 to 100
                     algorithm and is based, in the most                    
                     part, on the total number of plays                     
                     the track has had and how recent                       
                     those plays are. Generally speaking,                   
                     songs that are being played a lot                      
                     now will have a higher popularity                      
                     than songs that were played a lot in                   
                     the past.                                              

  Speechiness        Detects the presence of spoken words Float             0.0 to 1.0
                     in a track. The more exclusively                       
                     speech-like the recording (e.g. talk                   
                     show, audio book, poetry), the                         
                     closer to 1.0 the attribute value.                     
                     Values above 0.66 describe tracks                      
                     that are probably made entirely of                     
                     spoken words. Values between 0.33                      
                     and 0.66 describe tracks that may                      
                     contain both music and speech,                         
                     either in sections or layered,                         
                     including such cases as rap music.                     
                     Values below 0.33 most likely                          
                     represent music and other                              
                     non-speech-like tracks.                                

  Tempo              The overall estimated tempo of a     Float             0.0 to 1.0
                     track in beats per minute (BPM). In                    
                     musical terminology, tempo is the                      
                     speed or pace of a given piece and                     
                     derives directly from the average                      
                     beat duration.                                         

  Valence            A measure describing the musical     Float             0.0 to 1.0
                     positiveness conveyed by a track.                      
                     Tracks with high valence sound more                    
                     positive (e.g. happy, cheerful,                        
                     euphoric), while tracks with low                       
                     valence sound more negative (e.g.                      
                     sad, depressed, angry).                                
  ------------------ ------------------------------------ ----------------- --------------

Data Preprocessing

Since the goal of our project was to see if a song's decade of release
could be determined via its audio features, our target variable of
"decade" needed to be engineered via the song's release date.
Thankfully, the Kaggle user who originally created this dataset added an
additional variable column for year, which we then converted to our
target decade variable utilizing floor division (floor dividing by 10,
then multiplying the result by 10). The original data pulled from the
Spotify Web API also provides boolean values ("true" and "false") for
the "explicit" variable, however, the Kaggle dataset creator had already
converted these to dummy values of 1 and 0, saving us an additional step
in the data preprocessing procedure. No other data cleaning was
performed on the overall dataset (not including scaling or bucketization
for certain models), as the dataset did not have any missing values or
improperly formatted variables.

Feature Selection

The purpose of our model was to see if a music's decade could be
predicted purely based on its audio features; that is, features that
contribute to explaining musical elements, such as:

-   **Mood:** Danceability, Valence, Energy, Tempo

-   **Properties:** Loudness, Speechiness, Instrumentalness

-   **Context:** Liveness, Acousticness

Three other categorical features, which could also contribute to
explaining the musical elements of a track were also included. These
were explicit, key and mode.

Training Methodology

All character or time based variables were eliminated from the dataset
including popularity. The Song ID was kept in the modeling data to
maintain continuity without information leakage. Once the necessary
dimensions were collected the data was split into training and testing
sets utilizing PySpark's 'randomSplit' function, with a 75/25 split. A
seed of 314 was utilized in the 'randomSplit' function to ensure
consistency across all data utilized for model training and testing.

Each model had access to all of the features that could be expressed as
continuous or binary variables. However, training metrics were used to
identify the optimal feature subset for each model.

Exploratory Data Analysis

To begin our initial data exploration, we first looked at the average
values for each feature by decade. We produced the following table which
gave us a high level understanding of which features were in the data
and how the values looked for each feature.

![](media/image1.png){width="6.380208880139983in"
height="2.3645833333333335in"}

After looking at the averages for each feature, we decided to look at
the trends for each feature over time. This graph along with the
previously mentioned table helped us determine what information each
feature should reflect as we began the model building process.

![](media/image2.png){width="4.963542213473316in"
height="2.8715365266841646in"}

![](media/image3.png){width="1.0416666666666666e-2in"
height="1.0416666666666666e-2in"}![](media/image3.png){width="1.0416666666666666e-2in"
height="1.0416666666666666e-2in"}![](media/image3.png){width="1.0416666666666666e-2in"
height="1.0416666666666666e-2in"}

Models

**Random Forest (RF):**

The first model that was trained was a Random Forest. It is important to
note that a Gradient Boosted Tree (GBT) may have been a more logical
place to start, but spark is limited in its offering because their GBT
algorithm only allows binary classification. Intuitively, the next best
alternative was Random Forest because it exists in the same
ensemble-tree modeling space just with a slightly different approach.

In terms of the decade classification problem outlined in the
introduction the nature of a Random Forest has many benefits. Most of
the benefits stem from the fact that RF are non-parametric models with
controllable complexity, that do not require the scaling of features.
These characteristics mean that RF can be used/optimized for predictive
accuracy, but they can also be exploited as analytical tools by dialing
back the complexity and looking at the features of each node to
understand the linked effects that the different dimensions have. In
regards to the decades problem, this means that one could use a RF to
identify the way that certain song attributes are indicative of
different eras in a highly interpretable manner.

**Decision Tree:**

The next model we trained was a simple Decision Tree. Like Random
Forest, Decision Trees do not require feature scaling and are highly
interpretable. They also handle categorical predictors and feature
interactions extremely efficiently, as well as being able to handle
multiclass classification within the Spark implementation.
Cross-validation, with K=5 and an evaluation metric of *accuracy*, was
utilized to tune the hyperparameters of *maxDepth* and *maxBin*. The
final, "best" model had a depth of 10 and 1,641 nodes.

**One vs. Rest Logistic Regression :**

One-vs-Rest classification model was then built utilizing logistic
regression as the base classifier. One-vs-Rest, also known as
One-vs-All, is a machine learning reduction used to perform multiclass
classification by turning the problem into a series of *k*-classes
binary classification problems, where one class is predicted against all
the remaining classes. The classifier predicts whether the label belongs
to the tested class or not, and final predictions are done by evaluating
each classifier, where the most confident classifier is output as the
predicted label. The data was scaled before the series of logistic
regressions was performed, and cross-validation, with K=5, was used to
tune the model hyperparameters.

**Multinomial Logistic Regression:**

While building the multinomial logistic regression model, we looked at
two possible scenarios to build our model for. Before exploring the
data, we wanted to attempt to predict how popular a song would be based
on the features within the dataset. However, we decided that with this
scenario, time of the song was not taken into consideration and
therefore historical popularity would not be as useful to predict future
popularity. With this in mind, we decided to build a multinomial
logistic regression model to attempt to predict which song a decade was
from using the features previously discussed. The accuracy of this model
came out to be .3105.

**Naive Bayes:**

The final model utilized to predict song decades was the Multinomial
Naive Bayes classifier. The model is based on Bayes Theorem and
leverages frequency-based probability estimates to make predictions. For
this reason Naive Bayes performs well in the case of categorical data.
As such, we applied Spark's Bucketizer transformer to discretize
features that were represented by continuous data. However, the model
assigned zero probabilities in instances where feature categories
existed within the test set but not in the training set thus skewing the
performance of the classification. The estimation error is attributed to
this occurrence as well as the discretisation strategy.

Results

After running each model, we discovered that the accuracies for each
model were lower than we had thought when we first started our data
exploration and initial hypothesis development. The Random Forest model
produced a final test accuracy of .3701 which was the highest accuracy
of the five models that we had run. The multinomial logistic regression
had the second highest test accuracy followed by the One vs. Rest
Logistic Regression model.

![](media/image4.png){width="4.715115923009624in"
height="4.223958880139983in"}

Had we been looking at a dataset with more categorical data to predict
which decade the songs were in, we believe the accuracies of the models
would have increased.

Future Work

As stated in the previous section, we would like to possibly bring in
more data that would contain a larger amount of categorical features to
improve the accuracies of the models. With more accurate models, we
could confidently begin to predict which decade a song originated in
based on the features such as its speechiness, acousticness, etc.

Obtaining more accurate models would allow us to start looking at
different trends in each decade and analyzing why a decade contained
songs with certain values for each feature.

We would also like to explore the 'developer' account that users are
able to create for Spotify. The idea of potentially being able to scrape
more data and lyrics from the application itself could allow us to not
only begin to predict which decade certain songs belong to, but also to
start exploring the text within the lyrics themselves to determine how
the words for each decade changed over time.

Conclusions

In conclusion, the accuracies of all of the models that we built were
relatively low to what we were expecting. With more categorical data as
well as exploring different data sources to supplement the data we used
in this analysis, we believe that we would be able to produce more
accurate models that would successfully predict which decade a song was
in.

[^1]: [[https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks]{.underline}](https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks)
