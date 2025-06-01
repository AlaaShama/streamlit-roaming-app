import streamlit as st
import plotly.express as px
import pandas as pd
import datetime
from sklearn.base import TransformerMixin, BaseEstimator
import joblib


import numpy as np


class SafeLogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, offset=1e-10):
        """
        Parameters
        ----------
        offset : float, default=1e-10
            Small constant to add to avoid log(0) or log(negative)
        """
        self.offset = offset
    
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]  # Note the trailing underscore (sklearn convention)
        return self

    def transform(self, X, y=None):
        assert self.n_features_in_ == X.shape[1]
        # Add offset to handle zeros and negatives, then take log
        return np.log(X + self.offset)
    
    # Optional: get feature names (if working with DataFrames)
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return [f"log_{i}" for i in range(self.n_features_in_)]
        return [f"log_{name}" for name in input_features]


# Usage example:
safe_log_transformer = SafeLogTransformer(offset=1e-10)
# --- Sidebar Navigation ---
st.sidebar.title("Retail Analytics Platform")
page = st.sidebar.radio("Select a page:", ["🏠 Home", "📊 EDA Dashboard", "🔮 Purchase Predictor"])

# --- Home Page ---
if page == "🏠 Home":
    st.title("Retail Purchase Prediction App")
    st.subheader("Welcome to our Retail Analytics Platform 🎉")
    st.markdown("""
        This platform provides insights and predictive analytics for retail purchases.
        
        **Use the sidebar** to navigate through the application:
        - 📊 **EDA Dashboard:** Explore our data visually
        - 🔮 **Purchase Predictor:** Predict the likelihood of a customer making a purchase
    """)

# --- EDA Dashboard Page ---
elif page == "📊 EDA Dashboard":
    st.title("📊 EDA Dashboard")
    st.subheader("Explore Your Retail Data")

    uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        import pandas as pd
        import plotly.express as px

        df = pd.read_csv(uploaded_file)

        st.write("### 🧾 Data Preview", df.head())
        st.write("### 📊 Summary Statistics", df.describe())
        st.write("### 🏷️ Column Info", df.dtypes)

        # Cast numeric columns with < 9 unique values to categorical
        for col in df.select_dtypes(include='number').columns:
            if df[col].nunique() < 10:
                df[col] = df[col].astype("object")

        # Function to convert RGB to HEX
        def rgb_to_hex(r, g, b):
            return '#{:02X}{:02X}{:02X}'.format(r, g, b)

        # Apply the function row-wise
        df['sec_color_hex'] = df.apply(
            lambda row: rgb_to_hex(row['rgb_r_sec_col'], row['rgb_g_sec_col'], row['rgb_b_sec_col']),
            axis=1
        )
        df['sec_color_hex'] = df.apply(
            lambda row: rgb_to_hex(row['rgb_r_sec_col'], row['rgb_g_sec_col'], row['rgb_b_sec_col']),
            axis=1
        )
    
        # Apply the function row-wise
        df['main_color_hex'] = df.apply(
            lambda row: rgb_to_hex(row['rgb_r_main_col'], row['rgb_g_main_col'], row['rgb_b_main_col']),
            axis=1
        )
        df.drop(columns = ['rgb_r_main_col' ,'rgb_g_main_col', 'rgb_b_main_col', 'rgb_r_sec_col', 'rgb_g_sec_col', 'rgb_b_sec_col'], inplace=True)
        df['unit_profit'] = df['current_price'] - df['cost']
        df['discount_tier'] = pd.cut(df['ratio'], 
                            bins=[0, 0.3, 0.5, 0.7, 1],
                            labels=['deep_discount', 'medium_discount', 
                                   'small_discount', 'no_discount'])
        # Categorical Insights
        st.subheader("🧩 Categorical Column Insights")
        cat_cols = df.select_dtypes(include='O').columns

        if len(cat_cols) == 0:
            st.info("No categorical columns found.")
        else:
            for col in cat_cols:
                st.markdown(f"### Distribution of **{col}**")
                nunique = df[col].nunique()

                if 'sales' in df.columns:
                    if nunique < 7:
                        dff = df.groupby(col)[['sales']].count().reset_index().sort_values(by='sales', ascending=False)
                        fig = px.pie(dff, names=col, values='sales', title=f"{col} Distribution (Pie)")
                        st.plotly_chart(fig, use_container_width=True)
                    elif nunique < 50:
                        dff = df[col].value_counts().reset_index()
                        dff.columns = [col, 'count']
                        fig = px.bar(dff, x=col, y='count', title=f"{col} Distribution (Bar)")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        dff = df[col].value_counts().head(10).reset_index()
                        dff.columns = [col, 'count']
                        fig = px.bar(dff, x=col, y='count', title=f"Top 10 {col} Categories")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Column 'sales' not found for analysis of {col}")

        # Trend Analysis for Numeric Features
        st.subheader("📈 Numeric Feature Trend Analysis")

        num_cols = df.select_dtypes(include='number').columns.tolist()
        time_col = None

        # Choose a suitable time column for trend analysis
        for col in ['retailweek', 'month', 'year']:
            if col in df.columns:
                time_col = col
                break

        if time_col is None:
            st.warning("No time-based column (e.g. 'retailweek', 'month', or 'year') found for trend analysis.")
        else:
            st.info(f"Using **{time_col}** as the time axis.")
            for col in num_cols:
                if col != time_col:
                    trend_df = df.groupby(time_col)[col].mean().reset_index()
                    fig = px.line(trend_df, x=time_col, y=col, title=f"{col} Over Time")
                    st.plotly_chart(fig, use_container_width=True)

        # Sales by Label for Promo1 and Promo2
        st.subheader("🛍️ Sales by Label under Promotions")

        if all(col in df.columns for col in ['sales', 'label', 'promo1', 'promo2']):
            
            # Ensure promo1 and promo2 are categorical
            df['promo1'] = df['promo1'].astype(str)
            df['promo2'] = df['promo2'].astype(str)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📌 Promo1 Effect")
                promo1_df = df.groupby(['label', 'promo1'])['sales'].sum().reset_index()
                fig = px.bar(promo1_df, x='label', y='sales', color='promo1', barmode='group',
                            title="Sales by Label and Promo1 Status")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### 📌 Promo2 Effect")
                promo2_df = df.groupby(['label', 'promo2'])['sales'].sum().reset_index()
                fig = px.bar(promo2_df, x='label', y='sales', color='promo2', barmode='group',
                            title="Sales by Label and Promo2 Status")
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Columns 'sales', 'label', 'promo1', and/or 'promo2' not found in the dataset.")

    else:
        st.info("Please upload a CSV file to begin exploring.")



# --- Purchase Predictor Page ---
elif page == "🔮 Purchase Predictor":
    st.title("🔮 Purchase Predictor")
    st.subheader("Predict Customer Purchase Probability")

    import streamlit as st
    import pandas as pd
    import pickle

    # Load saved pipeline (preprocessor + model)
    with open("voting_pipeline.pkl", "rb") as f:
        model = joblib.load(f)
    # with open("E:\\e& task\\best_adaboost_model.pkl", "rb") as f:
          

    #       model = joblib.load(f)
    allowed_rgb_r_main_col = [205, 188, 138, 79, 139, 135, 181]
    allowed_rgb_g_main_col = [104, 238, 173, 140, 43, 148, 26, 206, 181, 137]
    allowed_rgb_b_main_col = [57, 104, 0, 149, 226, 205, 26, 250, 181, 137]

    allowed_rgb_r_sec_col = [205, 255, 164]
    allowed_rgb_g_sec_col = [155, 187, 211]
    allowed_rgb_b_sec_col = [155, 255, 238]




    st.markdown("Enter the following details to predict the purchase likelihood or sales value:")

    # Categorical Inputs
    country = st.selectbox("Country", ['Germany', 'Austria', 'France'])  # Add your list
    productgroup = st.selectbox("Product Group", ['SHOES' ,'SHORTS' ,'HARDWARE ACCESSORIES', 'SWEATSHIRTS'])
    category = st.selectbox("Category", ['TRAINING' ,'GOLF' ,'RUNNING' ,'RELAX CASUAL', 'FOOTBALL GENERIC', 'INDOOR'])
    gender = st.selectbox("Gender", ['women' ,'kids' ,'unisex' ,'men'])
    style = st.selectbox("Style", ['slim', 'regular', 'wide'])  # Add real values
    sizes = st.selectbox("Sizes", ['xxs,xs,s,m,l,xl,xxl' ,'xs,s,m,l,xl'])
    # sec_color_hex = st.selectbox("sec_color_hex", ['#FFBBFF', '#A4D3EE', '#CD9B9B'])
    # main_color_hex = st.selectbox("main_color_hex", ['#CD6839', '#BCEE68', '#CDAD00' ,'#CD8C95', '#8A2BE2', '#4F94CD', '#8B1A1A',
    # '#87CEFA', '#B5B5B5', '#8B8989'])
    promo1 = st.selectbox("Promo 1", [0, 1])
    promo2 = st.selectbox("Promo 2", [0, 1])
    rgb_r_main_col = st.selectbox("rgb_r_main_col", allowed_rgb_r_main_col)
    rgb_g_main_col = st.selectbox("rgb_g_main_col", allowed_rgb_g_main_col)
    rgb_b_main_col = st.selectbox("rgb_b_main_col", allowed_rgb_b_main_col)
    rgb_r_sec_col = st.selectbox("rgb_r_sec_col",allowed_rgb_r_sec_col)
    rgb_g_sec_col = st.selectbox("rgb_g_sec_col",allowed_rgb_g_sec_col)
    rgb_b_sec_col = st.selectbox("rgb_b_sec_col",allowed_rgb_b_sec_col)  




    article_options = ['YN8639', 'CF3238', 'WR9459', 'EF2771', 'LX1494', 'VF6733',
        'YK5786', 'CX1431', 'TK4862', 'ZM8792', 'LU3394', 'IW7978',
        'XG3252', 'UM7314', 'XG6449', 'ZJ5718', 'LI3529', 'KO9295',
        'GJ5184', 'UJ4517', 'BU9681', 'CR8478', 'VK4838', 'AA7884',
        'BY9685', 'UX6851', 'FF7283', 'CB8861', 'FE4648', 'BC1489',
        'PQ6953', 'WO1329', 'LH8921', 'TS8795', 'LI6472', 'CA2199',
        'WK5365', 'QS5396', 'TM4166', 'BF7459', 'JY1298', 'MP6772',
        'AZ5221', 'MR4948', 'UD3728', 'VT7698', 'DI9187', 'OZ8992',
        'UB1117', 'TN5256', 'BE9148', 'FU5676', 'ST3419', 'ZK4922',
        'FJ2121', 'KT8964', 'RO5412', 'BS7795', 'QP2819', 'GW8244',
        'AP5568', 'PP8845', 'GC8114', 'MJ2618', 'ZC7213', 'GD2286',
        'RN5619', 'EN9438', 'PZ7731', 'LR5226', 'BM9116', 'LD8468',
        'ML2223', 'NQ1161', 'GT5685', 'QU7755', 'GA4832', 'HM5731',
        'EH5694', 'ZV2187', 'KF6572', 'IR3275', 'MO9371', 'AA8941',
        'PN1714', 'LX5583', 'IA4131', 'PV1343', 'WP4135', 'CY6963',
        'VY6942', 'FB5424', 'PQ6773', 'PW6278', 'VA9789', 'IF7337',
        'XB1815', 'VL9749', 'CC8861', 'DM6271', 'ZZ2466', 'QX5316',
        'OP1184', 'ON6325', 'PF5685', 'JM7648', 'ZU2733', 'OK8155',
        'EB5477', 'RP9222', 'VY8356', 'ZX8794', 'FP2228', 'RN4195',
        'YR2438', 'ZR3493', 'RJ3725', 'LI5748', 'JB4241', 'WB3769',
        'BF7554', 'AX5971', 'ZR8112', 'XI5411', 'NB5887', 'UR7332',
        'HN7357', 'VR2932', 'WP4574', 'RE8863', 'MQ6248', 'FE6641',
        'JQ8333', 'AH6675', 'EA9617', 'VF7316', 'VJ8341', 'RJ5552',
        'GZ1752', 'DY1673', 'GZ5576', 'ZE6328', 'BE2333', 'QT2338',
        'ZW6694', 'XF4642', 'JI2453', 'RF2926', 'TX3691', 'VW9933',
        'XH6675', 'YX2167', 'HB1693', 'CB4942', 'AQ1643', 'XO7333',
        'SF1988', 'LB9256', 'PQ6379', 'SC5839', 'DZ3492', 'XC9518',
        'DH6848', 'VW8489', 'FP7124', 'BI5643', 'AR1923', 'AU7641',
        'YI3589', 'QB1247', 'JW4878', 'ET7242', 'RF6397', 'DG7643',
        'VX8496', 'WM7783', 'QL6154', 'ON9331', 'LC1964', 'DW8683',
        'MM4542', 'KL1526', 'SP6977', 'GL8661', 'PV4787', 'CK7156',
        'BR3179', 'CH6937', 'PL6969', 'FK7423', 'ZK3537', 'XU9926',
        'HN6759', 'HZ9888', 'WL2581', 'KT9618', 'PQ4964', 'KJ9185',
        'LU6658', 'FK6357', 'MW9292', 'QK7994', 'BJ4373', 'WB3723',
        'SL9748', 'ZU5523', 'WV8337', 'OE7548', 'YX1723', 'QO5375',
        'VT1698', 'XG6147', 'KT2132', 'RF6881', 'AO8265', 'JC5886',
        'KI5716', 'NW3584', 'TX1463', 'FS5149', 'TC9631', 'CA2479',
        'DB3258', 'TW8762', 'QS1816', 'JA4544', 'MG2169', 'VK5535',
        'VT3516', 'QT7325', 'OJ4847', 'GR3986', 'YG9479', 'FG2965',
        'QO8312', 'VS6613', 'VU8833', 'PE2872', 'JG1582', 'BF9848',
        'JK5796', 'RX4112', 'AN4895', 'NT3648', 'ZE9366', 'WF4276',
        'FE2938', 'XR5464', 'IH1672', 'FE6662', 'MK5273', 'RC5832',
        'AT7497', 'AD9697', 'FO4538', 'ZS4134', 'TX8432', 'EC5317',
        'PU1185', 'OY4474', 'NH7643', 'LT4238', 'AC7347', 'ON6494',
        'WC1828', 'GB6449', 'MW3528', 'AL2298', 'HW7772', 'QM3774',
        'RN7483', 'FJ8179', 'HM8568', 'CJ4578', 'JC1565', 'AX5913',
        'EL6462', 'TN7113', 'EF6812', 'SO4773', 'EI1264', 'VM7772',
        'TO2769', 'UX6816', 'JG6384', 'NL2136', 'HZ4826', 'HJ9196',
        'YN2747', 'VH1588', 'RV9228', 'QC7465', 'HD1628', 'GT2628',
        'OC6355', 'IM2273', 'HQ3171', 'AF5746', 'BX8284', 'XK8557',
        'TS8911', 'KY7934', 'XS4279', 'VP7827', 'PT2992', 'VD4566',
        'MW7971', 'MA7179', 'KV6219', 'SA2925', 'ZI6739', 'XF4182',
        'NK4915', 'CF4856', 'OF8158', 'FY5273', 'FV6234', 'WU2517',
        'ZT1211', 'VS2118', 'XI2961', 'ON4163', 'NY6781', 'AZ6626',
        'GR1127', 'QV8877', 'SG5828', 'KZ9384', 'PE5968', 'RE3197',
        'HN7272', 'EN1199', 'ZO6398', 'WJ9718', 'MI6988', 'YV6825',
        'IB8671', 'SE2934', 'TA7629', 'WQ8254', 'OI4367', 'NJ3895',
        'SW4387', 'YR6479', 'GS4461', 'RE8165', 'MC3398', 'NY5159',
        'PY1913', 'TS8227', 'ZX2294', 'NE7168', 'EZ8648', 'JR8311',
        'XT5836', 'UG4425', 'QG3131', 'YS6935', 'BH9952', 'GP6821',
        'UG2991', 'DM6477', 'GG3324', 'GP3497', 'KF7243', 'SV7732',
        'KV2454', 'YS9175', 'QD9777', 'DD1361', 'YV7315', 'KE3772',
        'VY8476', 'XI2814', 'TJ1277', 'TR1972', 'SG6172', 'ZD3611',
        'PY1419', 'JP9274', 'XB3134', 'NK3982', 'XJ1725', 'GG8661',
        'WE4646', 'IQ1913', 'OW5968', 'WT9578', 'EM9513', 'TL9924',
        'BZ4828', 'RH5979', 'LY8874', 'YV2782', 'HQ9691', 'XN6238',
        'BC6932', 'AL9977', 'PC6383', 'AM4669', 'QB6977', 'LL3852',
        'WZ7972', 'UV9411', 'YL7926', 'OO1497', 'AJ7542', 'DK3634',
        'RS3662', 'EU1121', 'KJ7255', 'BI5591', 'PY2718', 'XU7827',
        'SJ4545', 'UN9356', 'VG1586', 'SH7883', 'XF3362', 'BZ8791',
        'KF7125', 'NH9366', 'EZ3428', 'SW2464', 'BW2758', 'OU2254',
        'FX1729', 'DW2429', 'TY9287', 'XK6846', 'JX7462', 'NM4424',
        'JY1726', 'BX9481', 'LL7287', 'QD2412', 'OT2311', 'VC4517',
        'IW8485', 'CL8759', 'JR7981', 'LG5858', 'JN4924', 'PW7632',
        'MD2664', 'ZF7765', 'RT6283', 'KI2338', 'TO8135', 'OA8258',
        'LX5774', 'XH3727', 'SW7987', 'AR4473', 'HU6228', 'CQ8153',
        'NS7357', 'EL3283', 'IL7684', 'MZ9561', 'VX6536', 'ZB7415',
        'NY5947', 'YD2684', 'OV5561', 'AA1821', 'OU5334', 'VE4993',
        'LD1896', 'CO7738', 'PV7587', 'PH9161', 'WB8526', 'IO7646',
        'RX1584', 'QO7834', 'PB1483']
    article = st.selectbox("Article", sorted(article_options), index=0, key="article")
    article_1 = st.selectbox("Article.1", ['OC6355' ,'AP5568' ,'CB8861', 'LI3529', 'GG8661' ,'TX1463', 'PC6383' ,'VT7698'
    'FG2965' ,'AC7347']) 
    # Numeric Inputs

    cost = st.number_input("Cost", min_value=0.0)
    sales = st.number_input("sales", min_value=0.0)

    # --- Calendar input for date ---
    selected_date = st.date_input("Select a date from calendar")



    # --- Extract week, month, season ---
    retail_week = selected_date.strftime("%Y-%U")  # %U: week number (Sunday as first day of week)
    month = selected_date.month

    def get_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Fall"

    season = get_season(month)

    st.write(f"Extracted Month: **{month}**, Season: **{season}**, Retail Week: **{retail_week}**")





    regular_price = st.number_input("Regular Price", min_value=0.0)
    current_price = st.number_input("Current Price", min_value=0.0)
    if regular_price != 0:
        price_ratio = current_price / regular_price
    else:
        price_ratio = 0  # or np.nan or some default

    st.write(f"Calculated Price Ratio: **{price_ratio:.2f}**")

    discount = 1-price_ratio
    st.write(f"Calculated discount: **{discount:.2f}**")

    unit_profit = current_price - cost
    st.write(f"Calculated Unit_profit: **{unit_profit:.2f}**")
    def discountation(row):

        if 0 < row['price_ratio'] < 1:
            return '1'
        else:
            return '0'


    # --- PREDICTION ---
    if st.button("Predict"):
        input_df = pd.DataFrame([{
            'country': country,
            'productgroup': productgroup,
            'category': category,
            'gender': gender,
            # 'sec_color_hex': sec_color_hex,
            # 'main_color_hex':main_color_hex,
            'sizes': sizes,
            'article': article,
            'article.1': article_1,
            'cost': cost,
            'sales':sales,
            'style': style,
            'promo1': str(promo1),  # If these were stored as str originally
            'promo2': str(promo2),
            'regular_price': regular_price,
            'current_price': current_price,
            "price_ratio": price_ratio,
            "retail_week": retail_week,
            # "month": month,
            # "season": season,
            "unit_profit":unit_profit,
            "rgb_r_main_col":rgb_r_main_col,
            "rgb_g_main_col":rgb_g_main_col,
            "rgb_b_main_col":rgb_b_main_col,
            "rgb_r_sec_col":rgb_r_sec_col,
            "rgb_g_sec_col":rgb_g_sec_col,
            "rgb_b_sec_col":rgb_b_sec_col,

            "discount":discount
            
        }])
        input_df['discounted'] = input_df.apply(discountation, axis=1)

        import joblib

        # model = joblib.load("voting_pipeline.pkl")  # Use raw string or forward slashes
        prediction = model.predict(input_df)  # This should now work


        # Predict using the loaded model (includes preprocessing)
        # prediction = model.predict(input_df)

        st.success(f"💡 Predicted Sales/Purchase Value: **{prediction[0]:.2f}**")
