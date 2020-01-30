# TMDB Box Office Prediction

It is very important to have accurate prediction in case of box office prediction in order to determine future trends. Advanced machine learning techniques helps in predicting the revenue of movies released. This study consists of different machine learning methods like Linear Regression, K-Nearest Neighbors, and XG Boost. It covers data preprocessing, data cleaning, feature extraction and selecting the valuable features for training and testing. This repository focuses on regression models. The application of the algorithms will be evaluated on the TMDB box office dataset.

The dataset contains information about the past movies and contains the following features:

id: Represents an integer identification number for each movie.
belongs_to_collection: Contains the TMDB Id, Collection Name, Movie Poster and Backdrop URL of a movie in JSON format.
budget: demonstrates the budget of each movie in dollars.
genres: Includes  all the Genres Name & TMDB Id in JSON Format. 
homepage: Includes the official homepage URL of a movie.
imdb_id: IMDB id of a movie ( string format).
original_language: Contains two-digit code of the original language, in which the movie was produced. For example: en = English, fr = french.
original_title: The original title of a movie. If the original title is in English,  tittle & Original title are the same.
overview: Brief summary of the movie.
 popularity: Popularity of the movie (float type).
poster_path: Poster path of a movie.
production_companies: All production company name and TMDB id in JSON format of a movie.
production_countries: Two-digit code and full name of the production company in JSON format.
release_date: Release date of a movie(mm/dd/yy format) .
 runtime: Total runtime of a movie in minutes ( integer).
spoken_languages: Two-digit code and full name of the spoken language.
 status: Represents if the movie released or rumoured.
tagline: Tagline of a movie.
title: English title of a movie.
 keywords: TMDB id and name of all the keywords in JSON format.
cast: All cast TMDB id, name, character name, gender (1 = Female, 2 = Male) in JSON format.
crew: Name, TMDB id, profile path of various kind of crew members job like Director, Writer, Art, Sound etc.
revenue: Total revenue of each  movie in dollars
