import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score



movies = pd.read_csv("movieRating/data.csv", encoding="latin-1")

# Clean 'Year' column
movies["Year"] = movies["Year"].str.replace(r'\(|\)', '', regex=True)
movies["Year"] = pd.to_numeric(movies["Year"], errors="coerce")

movies["Votes"] = movies["Votes"].str.replace(",", "")
movies["Votes"] = pd.to_numeric(movies["Votes"], errors="coerce")

movies['Duration'] = pd.to_numeric(movies['Duration'].str.replace(' min', ''), errors="coerce")

movies = movies.drop(["Name", "Duration"], axis=1)
movies = movies.dropna(subset=["Rating"])

movies[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']] = movies[[
    'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3'
]].fillna('Unknown')


#Features
movies['Rating_Per_Vote'] = movies['Rating'] / movies['Votes']
director_avg_rating = movies.groupby('Director')['Rating'].mean()
movies['Director_Avg_Rating'] = movies['Director'].map(director_avg_rating)

actor_year = movies[['Year', 'Rating', 'Actor 1', 'Actor 2', 'Actor 3']].copy()
actor_year = actor_year.melt(
    id_vars=['Year', 'Rating'],
    value_vars=['Actor 1', 'Actor 2', 'Actor 3'],
    var_name='Actor_Role',
    value_name= 'Actor'
)

actor_year_avg = actor_year.groupby(['Actor', 'Year'])['Rating'].mean().reset_index()
actor_year_avg.rename(columns={'Rating': 'Actor_Year_Avg_Rating'}, inplace=True)

for col in ['Actor 1', 'Actor 2', 'Actor 3']:
    movies = movies.merge(
        actor_year_avg,
        how='left',
        left_on=[col, 'Year'],
        right_on=['Actor', 'Year']
    )

    new_col = col.replace(" ", "") + '_Year_Avg_Rating'
    movies.rename(columns={'Actor_Year_Avg_Rating': new_col}, inplace=True)
    movies.drop(columns=['Actor'], inplace=True)
    movies[new_col] = movies[new_col].fillna(movies['Rating'].mean())  # Fallback for missing values


movies = movies.dropna(subset=['Year', 'Votes'])

movies['Year'] = movies['Year'].astype(int)
movies['Votes'] = movies['Votes'].astype(int)

movies = pd.get_dummies(movies, columns=["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"])

# my pc cant handle the  full data
movies_sample = movies.sample(frac=0.2, random_state=42)

# Train-test split
X = movies_sample.drop("Rating", axis=1)
y = movies_sample["Rating"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)


#Plotting

residuals = y_test - y_pred_rf

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

#Predicted VS Actual Ratings
sns.scatterplot(x=y_test, y=y_pred_rf, ax=axs[0])
axs[0].plot([0, 10], [0, 10], '--r', lw=2)  # ref line
axs[0].set_title("Predicted vs Actual Ratings")
axs[0].set_xlabel("Actual Rating")
axs[0].set_ylabel("Predicted Rating")

#Residuals
sns.scatterplot(x=y_pred_rf, y=residuals, ax=axs[1])
axs[1].axhline(0, color = 'r', linestyle= '--')
axs[1].set_title("Residual Plot")
axs[1].set_xlabel("Predicted Ratings")
axs[1].set_ylabel("Residuals")

#histogram
sns.histplot(residuals, kde = True, ax=axs[2])
axs[2].set_title("Histogram of Residuals")
axs[2].set_xlabel("Residuals")
axs[2].set_ylabel("Frequency")

plt.tight_layout()
plt.show()

