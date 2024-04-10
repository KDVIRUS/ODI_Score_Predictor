import streamlit as st
import pickle
import pandas as pd


teams = ['India', 'South Africa']

venues = ['Green Park', 'Reliance Stadium', 'The Wanderers Stadium',
       'Wankhede Stadium', 'P Sara Oval', 'Kingsmead',
       'Rajiv Gandhi International Stadium, Uppal',
       'Hagley Oval, Christchurch',
       'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium',
       "St George's Park", 'The Rose Bowl',
       'MA Chidambaram Stadium, Chepauk', 'SuperSport Park', 'Grace Road',
       'Senwes Park', 'Kennington Oval',
       'Vidarbha Cricket Association Stadium, Jamtha', 'Newlands',
       'Sardar Patel Stadium, Motera', 'New Wanderers Stadium',
       'Sawai Mansingh Stadium', 'Melbourne Cricket Ground',
       'The Wanderers Stadium, Johannesburg', 'Sophia Gardens']

pipe_1 = pickle.load(open('pipe_rfr.pkl', 'rb'))
pipe_2 = pickle.load(open('pipe_xgbr.pkl', 'rb'))

st.title('ODI Score Predictor')

batting_team = st.selectbox('Select Batting Team', sorted(teams))
venue = st.selectbox('Select Venue',sorted(venues))

col3, col4, col5, col6 = st.columns(4)

with col3:
    current_score = st.number_input('Current Score')
with col4:
    over = st.number_input('Current Over')
with col5:
    ball = st.number_input('Ball No.')
with col6:
    wickets = st.number_input('Wickets out')

last_five = st.number_input('Runs scored in last 5 overs')

if st.button('Predict Score'):
    balls_left = 300 - ((over * 6) + ball)
    wickets_left = 10 - wickets
    crr = current_score / ((over * 6) + ball) * 6

    input_df = pd.DataFrame(
     {'batting_team' : [batting_team], 'venue' : venue, 'wickets_left': [wickets], 'balls_left': [balls_left], 'current_score' : [current_score], 'crr': [crr], 'last_five': [last_five]})
    result_1 = pipe_1.predict(input_df)
    result_2 = pipe_2.predict(input_df)
    st.header("Predicted Score by Random Forest Regressor- " + str(int(result_1[0])))
    st.header("Predicted Score by XG Boost Regressor- " + str(int(result_2[0])))

