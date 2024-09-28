# COMPAS-classifier.

{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "uP8S4XrixYdI"
      },
      "source": [
        "# Data Science Project\n",
        "\n",
        "By Muhammad Sanaullah KAYANI, Ifechukwu EJIOFOR, and Yizhou MA\n",
        "\n",
        "## Project Goal\n",
        "\n",
        "This project has three parts:\n",
        "\n",
        "- COMPAS scores have been shown to have biases against certain racial groups. Analyze the dataset to highlight these biases.  \n",
        "\n",
        "- Based on the features in the COMPAS dataset, train classifiers to predict who will re-offend.  Study if your classifiers are more or less fair than the COMPAS classifier.\n",
        "\n",
        "- Build a fair classifier. Is excluding the race from the feature set enough?"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Part 1: Dataset Exploration\n",
        "\n",
        "Analyse the dataset and do some initial statistics.\n",
        "\n",
        "#### __Download the dataset__\n",
        "\n",
        "We first need to download the dataset. *(See code cell below)*"
      ],
      "metadata": {
        "id": "KIE4tUo4EGrv"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 109,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3hgcObBVxYds",
        "outputId": "35426171-0790-473f-ae01-434ff7683bd3"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Looking for file 'compas-scores-two-years.csv' in the current directory...\n",
            "File found in current directory..\n"
          ]
        }
      ],
      "source": [
        "import os\n",
        "import urllib\n",
        "import random\n",
        "import numpy as np\n",
        "\n",
        "RANDOM_SEED = 1234\n",
        "random.seed(RANDOM_SEED)\n",
        "np.random.seed(RANDOM_SEED)\n",
        "\n",
        "def download_dataset(filename):\n",
        "    files = os.listdir(\".\") # get the current directory listing\n",
        "    print(f\"Looking for file '{filename}' in the current directory...\")\n",
        "    if filename not in files:\n",
        "        print(f\"'{filename}' not found! Downloading from GitHub...\")\n",
        "        addr = \"https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv\"\n",
        "        response = urllib.request.urlopen(addr)\n",
        "        data = response.read()\n",
        "        fileOut = open(filename, \"wb\")\n",
        "        fileOut.write(data)\n",
        "        fileOut.close()\n",
        "        print(f\"'{filename}' download and saved locally..\")\n",
        "    else:\n",
        "        print(\"File found in current directory..\")\n",
        "\n",
        "download_dataset(\"compas-scores-two-years.csv\")"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#### __Load dataset and clean it up__\n",
        "\n",
        "Next, we need to load the dataset and clean it up.\n",
        "\n",
        "__Load the dataset__\n",
        "\n",
        "- The dataset contains data on how many convicts? *(See code cell below)*\n",
        "\n",
        "- What are the features the dataset contains? *(See code cell below)*"
      ],
      "metadata": {
        "id": "-ZPkFb55x0Xn"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "\n",
        "# Load the dataset from the csv file\n",
        "dataset = pd.read_csv(\"compas-scores-two-years.csv\")\n",
        "\n",
        "# Print the number of convicts in the dataset\n",
        "print(\"The dataset contains data on\", dataset.shape[0], \"convicts.\")\n",
        "\n",
        "# Print the features that the dataset contains\n",
        "print(\"\\nBelow are the features that the dataset contains:\")\n",
        "print(dataset.columns)\n",
        "\n",
        "# Print the first few rows of the dataset\n",
        "pd.set_option(\"display.max_columns\", None)\n",
        "print(\"\\nBelow are the first few rows of the dataset:\")\n",
        "dataset.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 678
        },
        "id": "1ka0hotZy8J4",
        "outputId": "5e47bb18-0819-467c-97e1-bdaff3aacdcd"
      },
      "execution_count": 110,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "The dataset contains data on 7214 convicts.\n",
            "\n",
            "Below are the features that the dataset contains:\n",
            "Index(['id', 'name', 'first', 'last', 'compas_screening_date', 'sex', 'dob',\n",
            "       'age', 'age_cat', 'race', 'juv_fel_count', 'decile_score',\n",
            "       'juv_misd_count', 'juv_other_count', 'priors_count',\n",
            "       'days_b_screening_arrest', 'c_jail_in', 'c_jail_out', 'c_case_number',\n",
            "       'c_offense_date', 'c_arrest_date', 'c_days_from_compas',\n",
            "       'c_charge_degree', 'c_charge_desc', 'is_recid', 'r_case_number',\n",
            "       'r_charge_degree', 'r_days_from_arrest', 'r_offense_date',\n",
            "       'r_charge_desc', 'r_jail_in', 'r_jail_out', 'violent_recid',\n",
            "       'is_violent_recid', 'vr_case_number', 'vr_charge_degree',\n",
            "       'vr_offense_date', 'vr_charge_desc', 'type_of_assessment',\n",
            "       'decile_score.1', 'score_text', 'screening_date',\n",
            "       'v_type_of_assessment', 'v_decile_score', 'v_score_text',\n",
            "       'v_screening_date', 'in_custody', 'out_custody', 'priors_count.1',\n",
            "       'start', 'end', 'event', 'two_year_recid'],\n",
            "      dtype='object')\n",
            "\n",
            "Below are the first few rows of the dataset:\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   id                name   first         last compas_screening_date   sex  \\\n",
              "0   1    miguel hernandez  miguel    hernandez            2013-08-14  Male   \n",
              "1   3         kevon dixon   kevon        dixon            2013-01-27  Male   \n",
              "2   4            ed philo      ed        philo            2013-04-14  Male   \n",
              "3   5         marcu brown   marcu        brown            2013-01-13  Male   \n",
              "4   6  bouthy pierrelouis  bouthy  pierrelouis            2013-03-26  Male   \n",
              "\n",
              "          dob  age          age_cat              race  juv_fel_count  \\\n",
              "0  1947-04-18   69  Greater than 45             Other              0   \n",
              "1  1982-01-22   34          25 - 45  African-American              0   \n",
              "2  1991-05-14   24     Less than 25  African-American              0   \n",
              "3  1993-01-21   23     Less than 25  African-American              0   \n",
              "4  1973-01-22   43          25 - 45             Other              0   \n",
              "\n",
              "   decile_score  juv_misd_count  juv_other_count  priors_count  \\\n",
              "0             1               0                0             0   \n",
              "1             3               0                0             0   \n",
              "2             4               0                1             4   \n",
              "3             8               1                0             1   \n",
              "4             1               0                0             2   \n",
              "\n",
              "   days_b_screening_arrest            c_jail_in           c_jail_out  \\\n",
              "0                     -1.0  2013-08-13 06:03:42  2013-08-14 05:41:20   \n",
              "1                     -1.0  2013-01-26 03:45:27  2013-02-05 05:36:53   \n",
              "2                     -1.0  2013-04-13 04:58:34  2013-04-14 07:02:04   \n",
              "3                      NaN                  NaN                  NaN   \n",
              "4                      NaN                  NaN                  NaN   \n",
              "\n",
              "   c_case_number c_offense_date c_arrest_date  c_days_from_compas  \\\n",
              "0  13011352CF10A     2013-08-13           NaN                 1.0   \n",
              "1  13001275CF10A     2013-01-26           NaN                 1.0   \n",
              "2  13005330CF10A     2013-04-13           NaN                 1.0   \n",
              "3  13000570CF10A     2013-01-12           NaN                 1.0   \n",
              "4  12014130CF10A            NaN    2013-01-09                76.0   \n",
              "\n",
              "  c_charge_degree                   c_charge_desc  is_recid  r_case_number  \\\n",
              "0               F    Aggravated Assault w/Firearm         0            NaN   \n",
              "1               F  Felony Battery w/Prior Convict         1  13009779CF10A   \n",
              "2               F           Possession of Cocaine         1  13011511MM10A   \n",
              "3               F          Possession of Cannabis         0            NaN   \n",
              "4               F           arrest case no charge         0            NaN   \n",
              "\n",
              "  r_charge_degree  r_days_from_arrest r_offense_date  \\\n",
              "0             NaN                 NaN            NaN   \n",
              "1            (F3)                 NaN     2013-07-05   \n",
              "2            (M1)                 0.0     2013-06-16   \n",
              "3             NaN                 NaN            NaN   \n",
              "4             NaN                 NaN            NaN   \n",
              "\n",
              "                 r_charge_desc   r_jail_in  r_jail_out  violent_recid  \\\n",
              "0                          NaN         NaN         NaN            NaN   \n",
              "1  Felony Battery (Dom Strang)         NaN         NaN            NaN   \n",
              "2  Driving Under The Influence  2013-06-16  2013-06-16            NaN   \n",
              "3                          NaN         NaN         NaN            NaN   \n",
              "4                          NaN         NaN         NaN            NaN   \n",
              "\n",
              "   is_violent_recid vr_case_number vr_charge_degree vr_offense_date  \\\n",
              "0                 0            NaN              NaN             NaN   \n",
              "1                 1  13009779CF10A             (F3)      2013-07-05   \n",
              "2                 0            NaN              NaN             NaN   \n",
              "3                 0            NaN              NaN             NaN   \n",
              "4                 0            NaN              NaN             NaN   \n",
              "\n",
              "                vr_charge_desc  type_of_assessment  decile_score.1 score_text  \\\n",
              "0                          NaN  Risk of Recidivism               1        Low   \n",
              "1  Felony Battery (Dom Strang)  Risk of Recidivism               3        Low   \n",
              "2                          NaN  Risk of Recidivism               4        Low   \n",
              "3                          NaN  Risk of Recidivism               8       High   \n",
              "4                          NaN  Risk of Recidivism               1        Low   \n",
              "\n",
              "  screening_date v_type_of_assessment  v_decile_score v_score_text  \\\n",
              "0     2013-08-14     Risk of Violence               1          Low   \n",
              "1     2013-01-27     Risk of Violence               1          Low   \n",
              "2     2013-04-14     Risk of Violence               3          Low   \n",
              "3     2013-01-13     Risk of Violence               6       Medium   \n",
              "4     2013-03-26     Risk of Violence               1          Low   \n",
              "\n",
              "  v_screening_date  in_custody out_custody  priors_count.1  start   end  \\\n",
              "0       2013-08-14  2014-07-07  2014-07-14               0      0   327   \n",
              "1       2013-01-27  2013-01-26  2013-02-05               0      9   159   \n",
              "2       2013-04-14  2013-06-16  2013-06-16               4      0    63   \n",
              "3       2013-01-13         NaN         NaN               1      0  1174   \n",
              "4       2013-03-26         NaN         NaN               2      0  1102   \n",
              "\n",
              "   event  two_year_recid  \n",
              "0      0               0  \n",
              "1      1               1  \n",
              "2      0               1  \n",
              "3      0               0  \n",
              "4      0               0  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-e893caa5-1b4c-4bf1-b2ec-d1b01cc220bc\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>id</th>\n",
              "      <th>name</th>\n",
              "      <th>first</th>\n",
              "      <th>last</th>\n",
              "      <th>compas_screening_date</th>\n",
              "      <th>sex</th>\n",
              "      <th>dob</th>\n",
              "      <th>age</th>\n",
              "      <th>age_cat</th>\n",
              "      <th>race</th>\n",
              "      <th>juv_fel_count</th>\n",
              "      <th>decile_score</th>\n",
              "      <th>juv_misd_count</th>\n",
              "      <th>juv_other_count</th>\n",
              "      <th>priors_count</th>\n",
              "      <th>days_b_screening_arrest</th>\n",
              "      <th>c_jail_in</th>\n",
              "      <th>c_jail_out</th>\n",
              "      <th>c_case_number</th>\n",
              "      <th>c_offense_date</th>\n",
              "      <th>c_arrest_date</th>\n",
              "      <th>c_days_from_compas</th>\n",
              "      <th>c_charge_degree</th>\n",
              "      <th>c_charge_desc</th>\n",
              "      <th>is_recid</th>\n",
              "      <th>r_case_number</th>\n",
              "      <th>r_charge_degree</th>\n",
              "      <th>r_days_from_arrest</th>\n",
              "      <th>r_offense_date</th>\n",
              "      <th>r_charge_desc</th>\n",
              "      <th>r_jail_in</th>\n",
              "      <th>r_jail_out</th>\n",
              "      <th>violent_recid</th>\n",
              "      <th>is_violent_recid</th>\n",
              "      <th>vr_case_number</th>\n",
              "      <th>vr_charge_degree</th>\n",
              "      <th>vr_offense_date</th>\n",
              "      <th>vr_charge_desc</th>\n",
              "      <th>type_of_assessment</th>\n",
              "      <th>decile_score.1</th>\n",
              "      <th>score_text</th>\n",
              "      <th>screening_date</th>\n",
              "      <th>v_type_of_assessment</th>\n",
              "      <th>v_decile_score</th>\n",
              "      <th>v_score_text</th>\n",
              "      <th>v_screening_date</th>\n",
              "      <th>in_custody</th>\n",
              "      <th>out_custody</th>\n",
              "      <th>priors_count.1</th>\n",
              "      <th>start</th>\n",
              "      <th>end</th>\n",
              "      <th>event</th>\n",
              "      <th>two_year_recid</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>miguel hernandez</td>\n",
              "      <td>miguel</td>\n",
              "      <td>hernandez</td>\n",
              "      <td>2013-08-14</td>\n",
              "      <td>Male</td>\n",
              "      <td>1947-04-18</td>\n",
              "      <td>69</td>\n",
              "      <td>Greater than 45</td>\n",
              "      <td>Other</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>-1.0</td>\n",
              "      <td>2013-08-13 06:03:42</td>\n",
              "      <td>2013-08-14 05:41:20</td>\n",
              "      <td>13011352CF10A</td>\n",
              "      <td>2013-08-13</td>\n",
              "      <td>NaN</td>\n",
              "      <td>1.0</td>\n",
              "      <td>F</td>\n",
              "      <td>Aggravated Assault w/Firearm</td>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>Risk of Recidivism</td>\n",
              "      <td>1</td>\n",
              "      <td>Low</td>\n",
              "      <td>2013-08-14</td>\n",
              "      <td>Risk of Violence</td>\n",
              "      <td>1</td>\n",
              "      <td>Low</td>\n",
              "      <td>2013-08-14</td>\n",
              "      <td>2014-07-07</td>\n",
              "      <td>2014-07-14</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>327</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>3</td>\n",
              "      <td>kevon dixon</td>\n",
              "      <td>kevon</td>\n",
              "      <td>dixon</td>\n",
              "      <td>2013-01-27</td>\n",
              "      <td>Male</td>\n",
              "      <td>1982-01-22</td>\n",
              "      <td>34</td>\n",
              "      <td>25 - 45</td>\n",
              "      <td>African-American</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>-1.0</td>\n",
              "      <td>2013-01-26 03:45:27</td>\n",
              "      <td>2013-02-05 05:36:53</td>\n",
              "      <td>13001275CF10A</td>\n",
              "      <td>2013-01-26</td>\n",
              "      <td>NaN</td>\n",
              "      <td>1.0</td>\n",
              "      <td>F</td>\n",
              "      <td>Felony Battery w/Prior Convict</td>\n",
              "      <td>1</td>\n",
              "      <td>13009779CF10A</td>\n",
              "      <td>(F3)</td>\n",
              "      <td>NaN</td>\n",
              "      <td>2013-07-05</td>\n",
              "      <td>Felony Battery (Dom Strang)</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>1</td>\n",
              "      <td>13009779CF10A</td>\n",
              "      <td>(F3)</td>\n",
              "      <td>2013-07-05</td>\n",
              "      <td>Felony Battery (Dom Strang)</td>\n",
              "      <td>Risk of Recidivism</td>\n",
              "      <td>3</td>\n",
              "      <td>Low</td>\n",
              "      <td>2013-01-27</td>\n",
              "      <td>Risk of Violence</td>\n",
              "      <td>1</td>\n",
              "      <td>Low</td>\n",
              "      <td>2013-01-27</td>\n",
              "      <td>2013-01-26</td>\n",
              "      <td>2013-02-05</td>\n",
              "      <td>0</td>\n",
              "      <td>9</td>\n",
              "      <td>159</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>4</td>\n",
              "      <td>ed philo</td>\n",
              "      <td>ed</td>\n",
              "      <td>philo</td>\n",
              "      <td>2013-04-14</td>\n",
              "      <td>Male</td>\n",
              "      <td>1991-05-14</td>\n",
              "      <td>24</td>\n",
              "      <td>Less than 25</td>\n",
              "      <td>African-American</td>\n",
              "      <td>0</td>\n",
              "      <td>4</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>4</td>\n",
              "      <td>-1.0</td>\n",
              "      <td>2013-04-13 04:58:34</td>\n",
              "      <td>2013-04-14 07:02:04</td>\n",
              "      <td>13005330CF10A</td>\n",
              "      <td>2013-04-13</td>\n",
              "      <td>NaN</td>\n",
              "      <td>1.0</td>\n",
              "      <td>F</td>\n",
              "      <td>Possession of Cocaine</td>\n",
              "      <td>1</td>\n",
              "      <td>13011511MM10A</td>\n",
              "      <td>(M1)</td>\n",
              "      <td>0.0</td>\n",
              "      <td>2013-06-16</td>\n",
              "      <td>Driving Under The Influence</td>\n",
              "      <td>2013-06-16</td>\n",
              "      <td>2013-06-16</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>Risk of Recidivism</td>\n",
              "      <td>4</td>\n",
              "      <td>Low</td>\n",
              "      <td>2013-04-14</td>\n",
              "      <td>Risk of Violence</td>\n",
              "      <td>3</td>\n",
              "      <td>Low</td>\n",
              "      <td>2013-04-14</td>\n",
              "      <td>2013-06-16</td>\n",
              "      <td>2013-06-16</td>\n",
              "      <td>4</td>\n",
              "      <td>0</td>\n",
              "      <td>63</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>5</td>\n",
              "      <td>marcu brown</td>\n",
              "      <td>marcu</td>\n",
              "      <td>brown</td>\n",
              "      <td>2013-01-13</td>\n",
              "      <td>Male</td>\n",
              "      <td>1993-01-21</td>\n",
              "      <td>23</td>\n",
              "      <td>Less than 25</td>\n",
              "      <td>African-American</td>\n",
              "      <td>0</td>\n",
              "      <td>8</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>13000570CF10A</td>\n",
              "      <td>2013-01-12</td>\n",
              "      <td>NaN</td>\n",
              "      <td>1.0</td>\n",
              "      <td>F</td>\n",
              "      <td>Possession of Cannabis</td>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>Risk of Recidivism</td>\n",
              "      <td>8</td>\n",
              "      <td>High</td>\n",
              "      <td>2013-01-13</td>\n",
              "      <td>Risk of Violence</td>\n",
              "      <td>6</td>\n",
              "      <td>Medium</td>\n",
              "      <td>2013-01-13</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1174</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>6</td>\n",
              "      <td>bouthy pierrelouis</td>\n",
              "      <td>bouthy</td>\n",
              "      <td>pierrelouis</td>\n",
              "      <td>2013-03-26</td>\n",
              "      <td>Male</td>\n",
              "      <td>1973-01-22</td>\n",
              "      <td>43</td>\n",
              "      <td>25 - 45</td>\n",
              "      <td>Other</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>2</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>12014130CF10A</td>\n",
              "      <td>NaN</td>\n",
              "      <td>2013-01-09</td>\n",
              "      <td>76.0</td>\n",
              "      <td>F</td>\n",
              "      <td>arrest case no charge</td>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>Risk of Recidivism</td>\n",
              "      <td>1</td>\n",
              "      <td>Low</td>\n",
              "      <td>2013-03-26</td>\n",
              "      <td>Risk of Violence</td>\n",
              "      <td>1</td>\n",
              "      <td>Low</td>\n",
              "      <td>2013-03-26</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>1102</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-e893caa5-1b4c-4bf1-b2ec-d1b01cc220bc')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-e893caa5-1b4c-4bf1-b2ec-d1b01cc220bc button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-e893caa5-1b4c-4bf1-b2ec-d1b01cc220bc');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-232c461b-6ca6-491f-ad0d-b475f3cf9846\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-232c461b-6ca6-491f-ad0d-b475f3cf9846')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-232c461b-6ca6-491f-ad0d-b475f3cf9846 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "dataset"
            }
          },
          "metadata": {},
          "execution_count": 110
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "__Cleanup the data__\n",
        "\n",
        "- Are there missing values (NaN)? *Yes, there are missing values. (See code cell below)*\n",
        "- Are there outliers?\n"
      ],
      "metadata": {
        "id": "QoNy9w3S0Ggh"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Calculate and print the sum of missing values in every column\n",
        "missing_values = dataset.isnull().sum()\n",
        "print(\"Below is the number of missing values in every column:\")\n",
        "print(missing_values)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "g98_N9lP0Cxs",
        "outputId": "8e37a383-7529-448d-8f8e-35ca38992ca4"
      },
      "execution_count": 111,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Below is the number of missing values in every column:\n",
            "id                            0\n",
            "name                          0\n",
            "first                         0\n",
            "last                          0\n",
            "compas_screening_date         0\n",
            "sex                           0\n",
            "dob                           0\n",
            "age                           0\n",
            "age_cat                       0\n",
            "race                          0\n",
            "juv_fel_count                 0\n",
            "decile_score                  0\n",
            "juv_misd_count                0\n",
            "juv_other_count               0\n",
            "priors_count                  0\n",
            "days_b_screening_arrest     307\n",
            "c_jail_in                   307\n",
            "c_jail_out                  307\n",
            "c_case_number                22\n",
            "c_offense_date             1159\n",
            "c_arrest_date              6077\n",
            "c_days_from_compas           22\n",
            "c_charge_degree               0\n",
            "c_charge_desc                29\n",
            "is_recid                      0\n",
            "r_case_number              3743\n",
            "r_charge_degree            3743\n",
            "r_days_from_arrest         4898\n",
            "r_offense_date             3743\n",
            "r_charge_desc              3801\n",
            "r_jail_in                  4898\n",
            "r_jail_out                 4898\n",
            "violent_recid              7214\n",
            "is_violent_recid              0\n",
            "vr_case_number             6395\n",
            "vr_charge_degree           6395\n",
            "vr_offense_date            6395\n",
            "vr_charge_desc             6395\n",
            "type_of_assessment            0\n",
            "decile_score.1                0\n",
            "score_text                    0\n",
            "screening_date                0\n",
            "v_type_of_assessment          0\n",
            "v_decile_score                0\n",
            "v_score_text                  0\n",
            "v_screening_date              0\n",
            "in_custody                  236\n",
            "out_custody                 236\n",
            "priors_count.1                0\n",
            "start                         0\n",
            "end                           0\n",
            "event                         0\n",
            "two_year_recid                0\n",
            "dtype: int64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "jp-MarkdownHeadingCollapsed": true,
        "id": "ZLkqvlDKxYdu"
      },
      "source": [
        "\n",
        "- Does ProPublica mentions how to clean the data? *Yes, ProPublica mentions how to clean the data.*\n",
        "\n",
        "- What is the effect of the following function? *The effect of the following function is to clean the data according to the criteria mentioned by ProPublica. (See code cell below)*"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Print the shape of the dataset before cleaning\n",
        "print(\"Shape of dataset before cleaning is: \", dataset.shape)\n",
        "\n",
        "# Drop all rows where the \"days_b_screening_arrest\" column is null\n",
        "dataset = dataset.dropna(subset=[\"days_b_screening_arrest\"])\n",
        "\n",
        "# Clean the data according to the criteria mentioned by ProPublica\n",
        "dataset = dataset[\n",
        "    (dataset[\"days_b_screening_arrest\"] <= 30) &\n",
        "    (dataset[\"days_b_screening_arrest\"] >= -30) &\n",
        "    (dataset[\"is_recid\"] != -1) &\n",
        "    (dataset[\"c_charge_degree\"] != 'O') &\n",
        "    (dataset[\"score_text\"] != 'N/A') &\n",
        "    (dataset[\"c_days_from_compas\"] <= 6000)\n",
        "]\n",
        "\n",
        "dataset_fair_model = dataset #used later for buiding fair model in Part 3\n",
        "# Re-number the rows from 0 again\n",
        "dataset.reset_index(inplace=True, drop=True)\n",
        "\n",
        "# Print the shape of the dataset after cleaning\n",
        "print(\"Shape of dataset after cleaning is: \", dataset.shape)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "l-75jTNG28Ba",
        "outputId": "2f8cc592-4da7-4853-8c30-1a4855bd0f32"
      },
      "execution_count": 112,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Shape of dataset before cleaning is:  (7214, 53)\n",
            "Shape of dataset after cleaning is:  (6170, 53)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#### __Select relevant features from the dataset__\n",
        "\n",
        "Now, we select the features that we find relevant from the dataset. This enables us to remove missing values."
      ],
      "metadata": {
        "id": "3ajTrF0cMaIv"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Select the features that we find relevant from the dataset\n",
        "dataset = dataset[[\"sex\", \"dob\", \"age\", \"age_cat\", \"race\", \"juv_fel_count\", \"juv_misd_count\", \"juv_other_count\",\n",
        "                  \"priors_count\", \"c_charge_degree\", \"start\", \"end\", \"event\",\n",
        "                  \"decile_score\", \"score_text\", \"v_decile_score\", \"two_year_recid\", \"is_violent_recid\"]]\n",
        "\n",
        "# Now, to show that missing values have been removed, we calculate and print the sum of missing values in every column\n",
        "missing_values = dataset.isnull().sum()\n",
        "print(\"Below is the number of missing values in every column:\")\n",
        "print(missing_values)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "I_3SZr2NzNvx",
        "outputId": "bd6740c4-64fc-42ce-d4de-cc097ec02273"
      },
      "execution_count": 113,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Below is the number of missing values in every column:\n",
            "sex                 0\n",
            "dob                 0\n",
            "age                 0\n",
            "age_cat             0\n",
            "race                0\n",
            "juv_fel_count       0\n",
            "juv_misd_count      0\n",
            "juv_other_count     0\n",
            "priors_count        0\n",
            "c_charge_degree     0\n",
            "start               0\n",
            "end                 0\n",
            "event               0\n",
            "decile_score        0\n",
            "score_text          0\n",
            "v_decile_score      0\n",
            "two_year_recid      0\n",
            "is_violent_recid    0\n",
            "dtype: int64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#### __Basic analysis of demographics__\n",
        "\n",
        "- What are the different races present in the dataset? *(See code cell below)*"
      ],
      "metadata": {
        "id": "5BIB0qja4v9N"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"The different races present in the dataset are:\")\n",
        "print(dataset[\"race\"].unique())"
      ],
      "metadata": {
        "id": "OvyWcRLv4F3L",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "840dc4b2-0bca-41ac-9495-bb38f0dbed9b"
      },
      "execution_count": 114,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "The different races present in the dataset are:\n",
            "['Other' 'African-American' 'Caucasian' 'Hispanic' 'Asian'\n",
            " 'Native American']\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "- What is the number of people by age category? *(See code cell below)*"
      ],
      "metadata": {
        "id": "pfkPvRay5afy"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"The different number of people by age category are:\")\n",
        "print(dataset[\"age_cat\"].value_counts())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "BWR2_Erv5aEC",
        "outputId": "a6049f45-0706-4702-ebc7-ba3bb126986a"
      },
      "execution_count": 115,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "The different number of people by age category are:\n",
            "age_cat\n",
            "25 - 45            3532\n",
            "Less than 25       1347\n",
            "Greater than 45    1291\n",
            "Name: count, dtype: int64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "- What is the number of people by race? *(See code cell below)*"
      ],
      "metadata": {
        "id": "R0dA8Qce55-7"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"The number of people by race are:\")\n",
        "print(dataset[\"race\"].value_counts())"
      ],
      "metadata": {
        "id": "XyvFB85M55ka",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "a5ec1e4a-be66-47e4-cc25-c028b433e8cf"
      },
      "execution_count": 116,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "The number of people by race are:\n",
            "race\n",
            "African-American    3174\n",
            "Caucasian           2103\n",
            "Hispanic             508\n",
            "Other                343\n",
            "Asian                 31\n",
            "Native American       11\n",
            "Name: count, dtype: int64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "- What is the number of people by COMPAS score (`decile_score`)? *(See code cell below)*"
      ],
      "metadata": {
        "id": "_LWscY_76VQx"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"The number of people by COMPAS score (decile_score) are:\")\n",
        "print(dataset[\"decile_score\"].value_counts())"
      ],
      "metadata": {
        "id": "kCPySGiM6U58",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "a4bd739d-0f73-41e6-fbab-07af0c827964"
      },
      "execution_count": 117,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "The number of people by COMPAS score (decile_score) are:\n",
            "decile_score\n",
            "1     1284\n",
            "2      822\n",
            "4      666\n",
            "3      647\n",
            "5      582\n",
            "6      529\n",
            "7      496\n",
            "9      420\n",
            "8      420\n",
            "10     304\n",
            "Name: count, dtype: int64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "bLhG12QLxYdv"
      },
      "source": [
        "- What is the number of people by COMPAS risk category (`score_text`)? *(See code cell below)*"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"The number of people by COMPAS risk category (score_text) are:\")\n",
        "print(dataset[\"score_text\"].value_counts())"
      ],
      "metadata": {
        "id": "xKEN7BjH60Op",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "5ea970ef-e553-4d54-a30d-f3fc0a2a5100"
      },
      "execution_count": 118,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "The number of people by COMPAS risk category (score_text) are:\n",
            "score_text\n",
            "Low       3419\n",
            "Medium    1607\n",
            "High      1144\n",
            "Name: count, dtype: int64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#### __Basic investigations of gender and race bias in COMPAS scores__\n",
        "\n",
        "- What is the mean COMPAS score (`decile_score`) per race? *(See code cell below)*"
      ],
      "metadata": {
        "id": "tf1SVjQA_N57"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "dataset_grouped_by_race = dataset.groupby(\"race\")\n",
        "\n",
        "print(\"The mean COMPAS score (decile score) per race are:\")\n",
        "print(dataset_grouped_by_race[\"decile_score\"].mean())"
      ],
      "metadata": {
        "id": "xW-NZXIZ_Nkl",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "08140f0a-e052-4366-fb8e-a61a88fb479f"
      },
      "execution_count": 119,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "The mean COMPAS score (decile score) per race are:\n",
            "race\n",
            "African-American    5.278198\n",
            "Asian               2.838710\n",
            "Caucasian           3.635283\n",
            "Hispanic            3.387795\n",
            "Native American     6.454545\n",
            "Other               2.889213\n",
            "Name: decile_score, dtype: float64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "- What is the mean COMPAS score (`decile_score`) per gender? *(See code cell below)*"
      ],
      "metadata": {
        "id": "1hlavUicYEnH"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "dataset_grouped_by_gender = dataset.groupby(\"sex\")\n",
        "\n",
        "print(\"The mean COMPAS score (decile score) per gender are:\")\n",
        "print(dataset_grouped_by_gender[\"decile_score\"].mean())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MYaPnO39YFiZ",
        "outputId": "22a3ed65-155a-4f4f-b377-9ea5ff798d12"
      },
      "execution_count": 120,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "The mean COMPAS score (decile score) per gender are:\n",
            "sex\n",
            "Female    4.066440\n",
            "Male      4.502602\n",
            "Name: decile_score, dtype: float64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "- What is the distribution (histogram) of `decile_score` per race? *(See code cell below)*"
      ],
      "metadata": {
        "id": "G67lccW_Er5f"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Initialize an array to hold the colors of the histogram\n",
        "histogram_colors = np.array([\"black\", \"red\", \"green\", \"blue\", \"orange\", \"purple\"])\n",
        "\n",
        "# Sort the names of unique races into an array\n",
        "races = np.sort(dataset[\"race\"].unique())\n",
        "\n",
        "# For each race, i, get data about the race and plot histogram for the race\n",
        "for i in range(len(races)):\n",
        "    data_about_race_i = dataset[dataset['race'] == races[i]]\n",
        "    plt.hist(data_about_race_i['decile_score'], bins=None, alpha=0.3, color=histogram_colors[i], label=races[i])\n",
        "\n",
        "plt.xlabel(\"Decile Score\")\n",
        "plt.ylabel(\"Frequency\")\n",
        "plt.title(\"Histogram of decile scores per race\")\n",
        "plt.legend()\n",
        "plt.grid(True)\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "PDouqNnUOvQo",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 472
        },
        "outputId": "132fa0f0-e43f-46d4-be0f-cb4b58db8093"
      },
      "execution_count": 121,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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uVZERERE2qBEE5Rzc3OxceNGHD58GF5eXirPxFq6dKlagiMiIiIqrWIlO3fu3IGTkxOuXbuGJk2aAABu3rypVEcikagvOiIiIqJSKlay4+rqisTERBw7dgzAq8dDrFy5Eg4ODmUSHBEREVFpFWvOzptPNT9w4AAyMjLUGhARERGROpVognK+N5MfIiIiIm1TrGRHIpGozMnhHB0iIiLSZsWasyMIAgIDA8WHfWZlZeGLL75QuRrrzz//VF+ERERERKVQrGQnICBAaXngwIFqDYaIiIhI3YqV7ISFhZVVHERERERlolQTlImIiIi0HZMdIiIi0mlMdoiIiEinMdkhIiIincZkh4iIiHRaiZ56TkSldz71fIlfqwc9NJE2waXnl6CAQo1RvVtT66bluj0iotLiyA4RERHpNCY7REREpNN4Got0QmlOCRERkW7jyA4RERHpNCY7REREpNOY7BAREZFOY7JDREREOo3JDhEREek0JjtERESk03jpORERUQWyZ88eTYdQLIIgaDoEjuwQERGRbmOyQ0RERDqNyQ4RERHpNCY7REREpNOY7BAREZFOY7JDREREOo3JDhEREek0JjtERESk05jsEBERkU5jskNEREQ6TaPJTkhICD744ANYWlrC3t4evXr1QmxsrFKdrKwsBAUFwdbWFhYWFvD390dycrJSnYSEBHTr1g1mZmawt7fHlClTkJubW567QkRERFpKo8nOiRMnEBQUhLNnzyIyMhJyuRydOnVCRkaGWGfChAnYs2cPtm/fjhMnTuDhw4fo06ePuD4vLw/dunVDTk4Ozpw5g02bNiE8PBwzZ87UxC4RERGRltHog0APHjyotBweHg57e3tcunQJrVu3xvPnz7FhwwZs3rwZ7du3BwCEhYXB3d0dZ8+eRfPmzREREYEbN27g8OHDcHBwgKenJ+bNm4dp06Zh9uzZMDIy0sSuERERkZbQqqeeP3/+HABgY2MDALh06RLkcjl8fX3FOm5ubqhRowaioqLQvHlzREVFwcPDAw4ODmIdPz8/jBo1CtevX0fjxo1VtpOdnY3s7GxxOS0tDQAgl8shl8vVuk96OjAtKn8fdGFfdIUm+0QbnmCsbfKPCY+NdmB/aJf8flD392tx2tSaZEehUGD8+PFo2bIlGjRoAABISkqCkZERrK2tleo6ODggKSlJrPN6opO/Pn9d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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "- What is the distribution (histogram) of `decile_score` per gender? *(See code cell below)*"
      ],
      "metadata": {
        "id": "rZLjjNBRW3ag"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Initialize an array to hold the colors of the histogram\n",
        "histogram_colors = np.array([\"green\", \"blue\"])\n",
        "\n",
        "# Store the names of unique genders into an array\n",
        "genders = dataset[\"sex\"].unique()\n",
        "\n",
        "# For each gender, i, get data about the gender and plot histogram for the gender\n",
        "for i in range(len(genders)):\n",
        "    data_about_gender_i = dataset[dataset['sex'] == genders[i]]\n",
        "    plt.hist(data_about_gender_i['decile_score'], bins=None, alpha=0.3, color=histogram_colors[i], label=genders[i])\n",
        "\n",
        "plt.xlabel(\"Decile Score\")\n",
        "plt.ylabel(\"Frequency\")\n",
        "plt.title(\"Histogram of decile scores per gender\")\n",
        "plt.legend()\n",
        "plt.grid(True)\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 472
        },
        "id": "e9yNa6f3W7ms",
        "outputId": "d0bdcce7-b48f-444d-812c-284ae3e8122e"
      },
      "execution_count": 122,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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IErF0zvpmcffh7Lpmnh/qPo4HvqF46F/OCb6pabGozrL0+tAtGnTJnz77bfYsGED2rVrh7S0NEyePBkuLi4ICwursfXGxMRg3rx55doTExNhYWFRY+utS3Zk7KiwPSkpqZYroSfheOgXjof+4ZjoF22PR0FBQZX76nUgevvtt/HOO+8gJCQEANChQwdcuXIFMTExCAsLg5OTEwAgKysLzs7O0uuysrLQqVMnAICTkxOys7PVlltSUoK7d+9Kr3/czJkzMXXqVGk6NzcXTZo0gb+/P2xsbLS5iQCAhIwErS+zpg3wGKA2rVQqkZSUhP79+8PY2FhHVVEZjod+4XjoH46Jfqmp8Sg7wlMVeh2ICgoKYGCgfpqToaEhVCoVAMDNzQ1OTk5ITk6WAlBubi4OHTqECRMmAAB8fHyQk5OD1NRUeHl5AQB27doFlUqFrl27VrheU1NTmJqalms3NjaukQ+OwlCh9WXWtMreh5p6j0gzHA/9wvHQPxwT/aLt8ajOsvQ6EAUFBWHRokVo2rQp2rVrh2PHjmH58uV48803AQAKhQKTJ0/GwoUL4eHhATc3N8yaNQsuLi4IDg4GALRp0waBgYEYN24c1qxZA6VSiejoaISEhPAKMyIiIgKg54Fo5cqVmDVrFt566y1kZ2fDxcUF48ePx+zZs6U+M2bMQH5+PiIjI5GTk4OePXti586dMDMzk/p8++23iI6Ohq+vLwwMDDB8+HCsWLFCF5tEREREekivA5G1tTViY2MRGxtbaR+FQoH58+dj/vz5lfaxt7fHhg0baqBCIiIieh7o9X2IiIiIiGoDAxERERHJHgMRERERyR4DEREREckeAxERERHJHgMRERERyR4DEREREckeAxERERHJHgMRERERyR4DEREREckeAxERERHJHgMRERERyR4DEREREckeAxERERHJHgMRERERyR4DEREREckeAxERERHJHgMRERERyR4DEREREckeAxERERHJHgMRERERyR4DEREREckeAxERERHJHgMRERERyR4DEREREckeAxERERHJHgMRERERyR4DEREREckeAxERERHJHgMRERERyR4DEREREckeAxERERHJHgMRERERyR4DEREREckeAxERERHJHgMRERERyR4DEREREckeAxERERHJHgMRERERyR4DEREREckeAxERERHJHgMRERER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P8Pa2lo658fW1hbm5uawtbXFmDFjMHXqVNjb28PGxgb//ve/4ePjg27dugEA/P390bZtW7zxxhtYunQpMjMz8d577yEqKqrCvUBEREQkP3odiFavXg0A6NOnj1p7XFwcwsPDAQAfffQRDAwMMHz4cBQVFSEgIACffvqp1NfQ0BC//PILJkyYAB8fH1haWiIsLAzz58+vrc0gIiIiPafXgUiIp5/5b2ZmhlWrVmHVqlWV9mnWrBl27NihzdKIiIjoOaLX5xARERER1QYGIiIiIpI9vT5kRvrr8C71m10aKFTwdM5C6j4Hvb2NQJd+WU/vREREsqSfv7mIiIiIahEDEREREckeAxERERHJHgMRERERyR4DEREREckeAxERERHJHgMRERERyR4DEREREckeAxERERHJHgMRERERyR4DEREREcken2VGsvH489fqAj5/jYiodnAPEREREckeAxERERHJHgMRERERyR7PISIireK5WkRUFzEQEekxTcOFgUIFT+cspO5zgEpwRzAR0dPwm5KIiIhkj4GIiIiIZI+BiIiIiGSP5xARkezV9IngNXFOF08EJ9Iu7iEiIiIi2WMgIiIiItljICIiIiLZYyAiIiIi2WMgIiIiItljICIiIiLZYyAiIiIi2WMgIiIiItljICIiIiLZ452qiYjqoJq+u3ZN4R22SV9xDxERERHJHgMRERERyR4DEREREckeAxERERHJHgMRERERyR4DEREREckeAxERERHJHgMRERERyR5vzEhERLVGX24oaaBQwdM5C6n7HKAST943wJtJygP3EBEREZHscQ8RERHRE+jLXq3q4F6t6uMeIiIiIpI9BiIiIiKSPR4yIyIies7UtcN8ZSe567QGna6diIiISA8wEBEREZHsMRARERGR7DEQERERkewxEBEREZHsMRARERGR7DEQERERkewxEBEREZHsMRARERGR7MkqEK1atQqurq4wMzND165dcfjwYV2XRERERHpANoHo+++/x9SpUzFnzhz89ddf6NixIwICApCdna3r0oiIiEjHZBOIli9fjnHjxiEiIgJt27bFmjVrYGFhgS+//FLXpREREZGOySIQFRcXIzU1FX5+flKbgYEB/Pz8kJKSosPKiIiISB/I4mn3t2/fRmlpKRwd1Z/+6+joiP/973/l+hcVFaGoqEiavn//PgDg7t27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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "- The `two_year_recid` field records whether or not each person was re-arrested for a violent offense within two years, which is what COMPAS is trying to predict. How many people were re-arrested? *(See code cell below)*"
      ],
      "metadata": {
        "id": "AhnQiN_WExYI"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "print(dataset[\"two_year_recid\"].sum(), \"people were re-arrested.\")"
      ],
      "metadata": {
        "id": "QMgahe9FEq7P",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "d75d5477-0900-4253-fb53-c6120e2d1887"
      },
      "execution_count": 123,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "2809 people were re-arrested.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "- Compute the recidivism (i.e., people that got re-arrested) rates by race. *(See code cell below)*"
      ],
      "metadata": {
        "id": "rod0ApeWFlKa"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "dataset_grouped_by_race = dataset.groupby(\"race\")\n",
        "\n",
        "print(\"The number of people who got re-arrested by race are:\")\n",
        "print(dataset_grouped_by_race[\"two_year_recid\"].sum())"
      ],
      "metadata": {
        "id": "HiwjfbBhFlu7",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "cc9f5604-dbc9-4457-ce33-888267c15005"
      },
      "execution_count": 124,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "The number of people who got re-arrested by race are:\n",
            "race\n",
            "African-American    1661\n",
            "Asian                  8\n",
            "Caucasian            822\n",
            "Hispanic             189\n",
            "Native American        5\n",
            "Other                124\n",
            "Name: two_year_recid, dtype: int64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "- Compute the recidivism (i.e., people that got re-arrested) rates by gender. *(See code cell below)*"
      ],
      "metadata": {
        "id": "ezWw9tpkbTR3"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "dataset_grouped_by_gender = dataset.groupby(\"sex\")\n",
        "\n",
        "print(\"\\nThe number of people who got re-arrested by gender are:\")\n",
        "print(dataset_grouped_by_gender[\"two_year_recid\"].sum())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qYVCWpV0bUQk",
        "outputId": "d022beb2-506b-439f-f6ed-ad0eea9ca5f3"
      },
      "execution_count": 125,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "The number of people who got re-arrested by gender are:\n",
            "sex\n",
            "Female     413\n",
            "Male      2396\n",
            "Name: two_year_recid, dtype: int64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "- What is the accuracy of the COMPAS scores to predict recidivism? *(See code cell below)*"
      ],
      "metadata": {
        "id": "kovhSK67HIkZ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import accuracy_score\n",
        "\n",
        "# Get the ground truth\n",
        "ground_truth = dataset[\"two_year_recid\"]\n",
        "\n",
        "# Get the predictions given by COMPAS. Here, we will consider a \"score_text\" value of \"High\" as a positive prediction\n",
        "predictions = (dataset[\"score_text\"] == \"High\")\n",
        "\n",
        "# Calculate the accuracy\n",
        "accuracy = accuracy_score(ground_truth, predictions)\n",
        "\n",
        "# Print the accuracy\n",
        "print(\"The accuracy of COMPAS scores is:\", accuracy)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "spCbuyQ6HIQL",
        "outputId": "f9fa5b77-2fff-43ac-c534-7d0d6dbbf265"
      },
      "execution_count": 126,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "The accuracy of COMPAS scores is: 0.63387358184765\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "- Is the accuracy higher/lower if we look at particular races? *Yes, if we look at races, the accuracy is lower for African-Americans, but higher for other races. (See code cell below)*"
      ],
      "metadata": {
        "id": "QRRZQE2vIzwM"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import accuracy_score\n",
        "\n",
        "print(\"Accuracy by race:\")\n",
        "\n",
        "for current in np.sort(dataset[\"race\"].unique()):\n",
        "    data_about_current_race = dataset[dataset[\"race\"] == current]         # Get data about current race\n",
        "    ground_truth = data_about_current_race[\"two_year_recid\"]        # Get the ground truth for current race\n",
        "    predictions = (data_about_current_race[\"score_text\"] == \"High\") # Get the predictions given by COMPAS for current race\n",
        "    accuracy = accuracy_score(ground_truth, predictions)      # Calculate the accuracy for current race\n",
        "    print(current, \": \", accuracy)                                  # Print the accuracy for current race"
      ],
      "metadata": {
        "id": "bS4HLT1HIzIe",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "c911af44-37d9-4d47-ba23-8b17ca985d0d"
      },
      "execution_count": 127,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy by race:\n",
            "African-American :  0.6099558916194077\n",
            "Asian :  0.7741935483870968\n",
            "Caucasian :  0.6571564431764146\n",
            "Hispanic :  0.6417322834645669\n",
            "Native American :  0.7272727272727273\n",
            "Other :  0.685131195335277\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "- Is the accuracy higher/lower if we look at particular genders? *Yes, if we look at gender, the accuracy is lower for males and higher for females. (See code cell below)*"
      ],
      "metadata": {
        "id": "SdTEqzFHe7XP"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import accuracy_score\n",
        "\n",
        "print(\"Accuracy by gender:\")\n",
        "\n",
        "for current in dataset[\"sex\"].unique():\n",
        "    data_about_current_gender = dataset[dataset[\"sex\"] == current]          # Get data about current gender\n",
        "    ground_truth = data_about_current_gender[\"two_year_recid\"]        # Get the ground truth for current gender\n",
        "    predictions = (data_about_current_gender[\"score_text\"] == \"High\") # Get the predictions given by COMPAS for current gender\n",
        "    accuracy = accuracy_score(ground_truth, predictions)        # Calculate the accuracy for current gender\n",
        "    print(current, \": \", accuracy)                                    # Print the accuracy for current gender"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "H3J0M9JMe8Dt",
        "outputId": "b1fe08de-4b83-4597-9a4b-0aaf6a80c9d7"
      },
      "execution_count": 128,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy by gender:\n",
            "Male :  0.6214971977582066\n",
            "Female :  0.686541737649063\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "I0NTh232xYdv"
      },
      "source": [
        "- What about false positives and false negatives?\n",
        " - *The false-positive rates of Native Americans is the worst (highest) followed by African-Americans.*\n",
        " - *Similarly, the false-negative rates of Native Americans is the worst (lowest) followed by African-Americans.*\n",
        " - **These results indicate that the COMPAS classifier was biased. It had stronger errors for some races compared to the others**. (See code cell below)*"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 129,
      "metadata": {
        "id": "-3uUQdUHxYdx",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "628ad2a5-d3b7-47ed-a388-487076e45e73"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "African-American : \n",
            "False positive rate:  0.1394580304031725 , False negative rate:  0.6183022275737508 \n",
            "\n",
            "Asian : \n",
            "False positive rate:  0.043478260869565216 , False negative rate:  0.75 \n",
            "\n",
            "Caucasian : \n",
            "False positive rate:  0.047619047619047616 , False negative rate:  0.8029197080291971 \n",
            "\n",
            "Hispanic : \n",
            "False positive rate:  0.06269592476489028 , False negative rate:  0.8571428571428571 \n",
            "\n",
            "Native American : \n",
            "False positive rate:  0.16666666666666666 , False negative rate:  0.4 \n",
            "\n",
            "Other : \n",
            "False positive rate:  0.0136986301369863 , False negative rate:  0.8467741935483871 \n",
            "\n"
          ]
        }
      ],
      "source": [
        "for current in np.sort(dataset[\"race\"].unique()):\n",
        "    data_about_current_race = dataset[dataset[\"race\"] == current]                     # Get data about current race\n",
        "    ground_truth = data_about_current_race[\"two_year_recid\"]                    # Get the ground truth for current race\n",
        "    predictions = (data_about_current_race[\"score_text\"] == \"High\").astype(int) # Get the predictions given by COMPAS for current race\n",
        "    confusion_matrix =  pd.crosstab(ground_truth, predictions)            # Generate the confusion matrix\n",
        "\n",
        "    FP = confusion_matrix[1][0] # False positives\n",
        "    FN = confusion_matrix[0][1] # False negatives\n",
        "    TP = confusion_matrix[1][1] # True positives\n",
        "    TN = confusion_matrix[0][0] # True negatives\n",
        "\n",
        "    FPR = FP / (FP + TN) # False positive rate\n",
        "    FNR = FN / (FN + TP) # False negative rate\n",
        "\n",
        "    print(current, \": \")\n",
        "    print(\"False positive rate: \", FPR, \", False negative rate: \", FNR , \"\\n\")"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "- To visualise these results, we plot false-positive rates vs thresholds. *(See code cell below)*"
      ],
      "metadata": {
        "id": "ZjJicqRFTWRj"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import roc_curve\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Function to plot false positive rates vs thresholds of the COMPAS classifier\n",
        "def plot_false_positive_rate_vs_threshold_of_COMPAS_classifier(races):\n",
        "    for current in races:\n",
        "        data_about_current_race = dataset[dataset[\"race\"] == current]\n",
        "        ground_truth = data_about_current_race[\"two_year_recid\"]\n",
        "        predictions = data_about_current_race[\"decile_score\"] / data_about_current_race[\"decile_score\"].max()\n",
        "        FPR, TPR, thresholds = roc_curve(ground_truth, predictions)\n",
        "        plt.plot(thresholds, FPR, label=current)\n",
        "    plt.ylabel(\"False Positive Rate\")\n",
        "    plt.xlabel(\"Threshold\")\n",
        "    plt.title(\"False positive rate vs treshold of COMPAS classifier\")\n",
        "    plt.legend()\n",
        "    plt.show()\n",
        "\n",
        "# Get names of all unique races in the dataset\n",
        "races = dataset[\"race\"].unique()\n",
        "\n",
        "# Call the function to plot false positive rates vs thresholds of the COMPAS classifier\n",
        "plot_false_positive_rate_vs_threshold_of_COMPAS_classifier(races)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 472
        },
        "id": "dQRkPtKgThuN",
        "outputId": "8d34c897-87f7-4bd8-982a-89f5b4031251"
      },
      "execution_count": 130,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "From the above plot, Native Americans have the highest false positive rates, followed by African-Americans.\n",
        "\n",
        "However, the dataset contains very few Native-American data (in fact, only 5 of them), there is no sufficient data to make conclusions about Native Americans. Similarly, there is no sufficient data for Asian, Hispanic, and Other races."
      ],
      "metadata": {
        "id": "ZXQehBu40xj9"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        ":So from now on, to make analysis simpler, we will focus only on data of African-Americans and Caucasians (they have the most number of data in the COMPAS dataset).\n",
        "\n",
        "- See below an equivalent of the above plot but now showing only African-Americans and Caucasians. *(See code cell below)*"
      ],
      "metadata": {
        "id": "hNsmwXS93M-m"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "races = np.array([\"African-American\", \"Caucasian\"])\n",
        "plot_false_positive_rate_vs_threshold_of_COMPAS_classifier(races)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 472
        },
        "id": "jqRDVHX84PB8",
        "outputId": "0f0e3440-c312-4d91-ccb0-7b0a5f20ef61"
      },
      "execution_count": 131,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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CwtLtZ9OnTrRunVr3N3dWbRoUaptL168SMWKFdNM/Hx8fDJ1/L///puWLVtSqFAhnJ2dqVKlSobvb4C5c+fy0ksv4ePjg729PRUqVEjz0llm3qdP+3w92eYmKCiIMWPGAODt7W322UqrzU1CQgJjxoyhdOnS2NvbExgYyIgRI1J952b0XZXXSM2NFYqJieHmzZtm67y8vDh48CB79+6la9euFC1alPDwcKZPn06jRo04ffo0Tk5Oae4vMTGRkJAQEhISGDx4MH5+fly7do1169Zx9+5d3N3dAeMvoE8++YTOnTvTv39/oqOjmTp1Ki+++CJHjx7N1K/Gzp07ExQUxPjx49m/fz/ff/89d+7cYcGCBaYy/fv3Z/78+XTq1In333+fv//+m/Hjx3PmzBnTdfeoqCiaN2+Ot7c3I0eOxMPDg/DwcFauXJnusTt27Mg///zD4sWLmTx5sumXZMo/2cfVqFGDkiVLsmzZMnr37m322NKlSylUqBAhISEA3LhxgxdeeMH0wff29mbDhg28/vrrxMbGMnTo0Kc+L9u3b2fZsmUMGjQILy8v06/P7777jrZt29K9e3cSExNZsmQJr776KuvWraNVq1aAsdFg//79qVWrFm+++SaA6Z/Qs8TWpUsXxo4dS2RkJH5+fqb1u3fv5vr163Tt2hWALVu20K1bN5o0acKECRMAOHPmDHv27GHIkCFp7tvb25vp06fz9ttv06FDBzp27AhAlSpVTGWSk5MJCQmhfv36TJw40fT+HTBgAPPmzaNv3768++67hIWF8cMPP3D06FH27NmDra2tRe+PRYsWce/ePQYMGIBGo+Hrr7+mY8eOXLp0CVtbWwC2bt1KixYtKFmyJGPHjuXhw4dMnTqVevXqceTIkVS1BY8bPXo0n3/+OS1btqRly5YcOXKE5s2bk5iYmO42KRRFoW3btuzYsYPXX3+datWqsWnTJj744AOuXbvG5MmT8fb2ZuHChXzxxRfcv3+f8ePHA1C+fPk097l7925u377N0KFDM5Vc/P777zg4ONC5c+c0Hy9RogT169dn+/btPHz4EEdHR9NjTk5OtGvXjsWLF5suGR07doxTp07x888/c/z48aceHyApKYmrV69SuHBh07q///6bCxcuMHfuXOzs7OjYsSO//vor//vf/8y2LV68OPv27ePkyZNZuny/ZcsWWrdu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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Again, this plot shows clearly that the COMPAS classifier was biased: It gave African-Americans consistently larger false positive rates than it gave Caucasians."
      ],
      "metadata": {
        "id": "K3oY8YsI4_-g"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Part 2: Standard Classifiers\n",
        "\n",
        "Now, we move on to train our own standard classifiers and compare with the COMPAS classifier.\n",
        "\n",
        "#### __First, we train a logistic regression classifier__\n",
        "\n",
        "- We select the variables to use for the classifier. *(See code cell below)*"
      ],
      "metadata": {
        "id": "pWLgf7zEXtn2"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Select the dependent variable\n",
        "y = dataset[[\"two_year_recid\"]]\n",
        "\n",
        "# Select the independent variables\n",
        "X = dataset[[\"sex\", \"dob\", \"age\", \"race\", \"juv_fel_count\", \"juv_misd_count\", \"juv_other_count\", \"priors_count\",\n",
        "             \"c_charge_degree\", \"start\", \"end\", \"event\"]]"
      ],
      "metadata": {
        "id": "pg9jqn75qGOu"
      },
      "execution_count": 132,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "- We convert all non-numerical data to numerical data. *(See code cell below)*\n",
        "\n"
      ],
      "metadata": {
        "id": "7AUoZrXg8TfQ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Convert \"sex\", \"race\", and \"c_charge_degree\" to numerical data (dummies)\n",
        "X = pd.get_dummies(X, columns=[\"sex\", \"race\", \"c_charge_degree\"], dtype=int)\n",
        "\n",
        "# Convert \"dob\" to integer\n",
        "X[\"dob\"] = pd.to_datetime(X[\"dob\"]).astype(int)\n",
        "\n",
        "# Print the first few rows of X to confirm that all non-numerical data have correctly been converted to numerical data\n",
        "X.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 244
        },
        "id": "1YwJIIUM8diW",
        "outputId": "3187bded-b2de-47a9-e898-5e52cef03230"
      },
      "execution_count": 133,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                  dob  age  juv_fel_count  juv_misd_count  juv_other_count  \\\n",
              "0 -716601600000000000   69              0               0                0   \n",
              "1  380505600000000000   34              0               0                0   \n",
              "2  674179200000000000   24              0               0                1   \n",
              "3   51667200000000000   44              0               0                0   \n",
              "4  143769600000000000   41              0               0                0   \n",
              "\n",
              "   priors_count  start  end  event  sex_Female  sex_Male  \\\n",
              "0             0      0  327      0           0         1   \n",
              "1             0      9  159      1           0         1   \n",
              "2             4      0   63      0           0         1   \n",
              "3             0      1  853      0           0         1   \n",
              "4            14      5   40      1           0         1   \n",
              "\n",
              "   race_African-American  race_Asian  race_Caucasian  race_Hispanic  \\\n",
              "0                      0           0               0              0   \n",
              "1                      1           0               0              0   \n",
              "2                      1           0               0              0   \n",
              "3                      0           0               0              0   \n",
              "4                      0           0               1              0   \n",
              "\n",
              "   race_Native American  race_Other  c_charge_degree_F  c_charge_degree_M  \n",
              "0                     0           1                  1                  0  \n",
              "1                     0           0                  1                  0  \n",
              "2                     0           0                  1                  0  \n",
              "3                     0           1                  0                  1  \n",
              "4                     0           0                  1                  0  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-24c946e0-a021-4bc1-8f21-b8600f2e761a\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>dob</th>\n",
              "      <th>age</th>\n",
              "      <th>juv_fel_count</th>\n",
              "      <th>juv_misd_count</th>\n",
              "      <th>juv_other_count</th>\n",
              "      <th>priors_count</th>\n",
              "      <th>start</th>\n",
              "      <th>end</th>\n",
              "      <th>event</th>\n",
              "      <th>sex_Female</th>\n",
              "      <th>sex_Male</th>\n",
              "      <th>race_African-American</th>\n",
              "      <th>race_Asian</th>\n",
              "      <th>race_Caucasian</th>\n",
              "      <th>race_Hispanic</th>\n",
              "      <th>race_Native American</th>\n",
              "      <th>race_Other</th>\n",
              "      <th>c_charge_degree_F</th>\n",
              "      <th>c_charge_degree_M</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>-716601600000000000</td>\n",
              "      <td>69</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>327</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>380505600000000000</td>\n",
              "      <td>34</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>9</td>\n",
              "      <td>159</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>674179200000000000</td>\n",
              "      <td>24</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>4</td>\n",
              "      <td>0</td>\n",
              "      <td>63</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>51667200000000000</td>\n",
              "      <td>44</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>853</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>143769600000000000</td>\n",
              "      <td>41</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>14</td>\n",
              "      <td>5</td>\n",
              "      <td>40</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-24c946e0-a021-4bc1-8f21-b8600f2e761a')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-24c946e0-a021-4bc1-8f21-b8600f2e761a button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-24c946e0-a021-4bc1-8f21-b8600f2e761a');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-0971a313-464f-4859-a0c3-19a22f925ac3\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-0971a313-464f-4859-a0c3-19a22f925ac3')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-0971a313-464f-4859-a0c3-19a22f925ac3 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "X",
              "summary": "{\n  \"name\": \"X\",\n  \"rows\": 6170,\n  \"fields\": [\n    {\n      \"column\": \"dob\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 369861900085055936,\n        \"min\": -1584748800000000000,\n        \"max\": 885254400000000000,\n        \"num_unique_values\": 4828,\n        \"samples\": [\n          753408000000000000,\n          248140800000000000,\n          677548800000000000\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"age\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 11,\n        \"min\": 18,\n        \"max\": 96,\n        \"num_unique_values\": 65,\n        \"samples\": [\n          83,\n          80,\n          69\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"juv_fel_count\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 20,\n        \"num_unique_values\": 10,\n        \"samples\": [\n          5,\n          2,\n          4\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"juv_misd_count\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 13,\n        \"num_unique_values\": 10,\n        \"samples\": [\n          5,\n          1,\n          4\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"juv_other_count\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 9,\n        \"num_unique_values\": 9,\n        \"samples\": [\n          6,\n          1,\n          9\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"priors_count\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 4,\n        \"min\": 0,\n        \"max\": 38,\n        \"num_unique_values\": 36,\n        \"samples\": [\n          38,\n          15,\n          17\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"start\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 50,\n        \"min\": 0,\n        \"max\": 937,\n        \"num_unique_values\": 236,\n        \"samples\": [\n          80,\n          98,\n          175\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"end\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 400,\n        \"min\": 0,\n        \"max\": 1186,\n        \"num_unique_values\": 1089,\n        \"samples\": [\n          811,\n          1179,\n          376\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"event\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"sex_Female\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"sex_Male\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          0,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"race_African-American\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"race_Asian\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"race_Caucasian\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"race_Hispanic\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"race_Native American\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"race_Other\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          0,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"c_charge_degree_F\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          0,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"c_charge_degree_M\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 133
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "- We split the data into training and test sets while ensuring that all races are equally represented. *(See code cell below)*"
      ],
      "metadata": {
        "id": "_m7jigmtJsju"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=X[[\"race_African-American\", \"race_Asian\", \"race_Caucasian\", \"race_Hispanic\", \"race_Native American\", \"race_Other\"]])"
      ],
      "metadata": {
        "id": "86VUBLT3JtZh"
      },
      "execution_count": 134,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "- We fit the data into a logistic regression classifier, then we predict. *(See code cell below)*"
      ],
      "metadata": {
        "id": "stywXmC6HlMV"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.pipeline import Pipeline\n",
        "\n",
        "logistic_regression = Pipeline([(\"scaler\", StandardScaler()), (\"classifier\", LogisticRegression())])\n",
        "logistic_regression.fit(X_train, y_train[\"two_year_recid\"])\n",
        "predicted_probabilities = logistic_regression.predict_proba(X_test)"
      ],
      "metadata": {
        "id": "Zf_9H-es-fiU"
      },
      "execution_count": 135,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "- For the fairness definition, I think we should use the definition of false positive rate of the classifier for each demographic group.\n"
      ],
      "metadata": {
        "id": "lW808ltC76Hz"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "- Now, it's time to compare with the COMPAS classifier.\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "CqRyQA951Vfv"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import roc_curve\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "def plot_false_positive_rate_vs_threshold_of_standard_classifier(races, predicted_probabilities):\n",
        "    for current in races:\n",
        "        current_race = (X_test[\"race\" + \"_\" + current] == 1)\n",
        "        ground_truth = y_test[current_race][\"two_year_recid\"]\n",
        "        predictions = predicted_probabilities[current_race][:,1]\n",
        "        FPR, TPR, thresholds = roc_curve(ground_truth, predictions)\n",
        "        plt.plot(thresholds, FPR, label=current)\n",
        "    plt.ylabel(\"False Positive Rate\")\n",
        "    plt.xlabel(\"Threshold\")\n",
        "    plt.title(\"False positive rate vs treshold of standard classifier\")\n",
        "    plt.legend()\n",
        "    plt.show()\n",
        "\n",
        "races = np.array([\"African-American\", \"Caucasian\"])\n",
        "plot_false_positive_rate_vs_threshold_of_standard_classifier(races, predicted_probabilities)\n",
        "plot_false_positive_rate_vs_threshold_of_COMPAS_classifier(races)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 927
        },
        "id": "Q9MV9cISk4Rk",
        "outputId": "a8e655ea-ca4d-40e1-e03d-e70c970aa679"
      },
      "execution_count": 136,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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CwtLtZ9OnTrRunVr3N3dWbRoUaptL168SMWKFdNM/Hx8fDJ1/L///puWLVtSqFAhnJ2dqVKlSobvb4C5c+fy0ksv4ePjg729PRUqVEjz0llm3qdP+3w92eYmKCiIMWPGAODt7W322UqrzU1CQgJjxoyhdOnS2NvbExgYyIgRI1J952b0XZXXSM2NFYqJieHmzZtm67y8vDh48CB79+6la9euFC1alPDwcKZPn06jRo04ffo0Tk5Oae4vMTGRkJAQEhISGDx4MH5+fly7do1169Zx9+5d3N3dAeMvoE8++YTOnTvTv39/oqOjmTp1Ki+++CJHjx7N1K/Gzp07ExQUxPjx49m/fz/ff/89d+7cYcGCBaYy/fv3Z/78+XTq1In333+fv//+m/Hjx3PmzBnTdfeoqCiaN2+Ot7c3I0eOxMPDg/DwcFauXJnusTt27Mg///zD4sWLmTx5sumXZMo/2cfVqFGDkiVLsmzZMnr37m322NKlSylUqBAhISEA3LhxgxdeeMH0wff29mbDhg28/vrrxMbGMnTo0Kc+L9u3b2fZsmUMGjQILy8v06/P7777jrZt29K9e3cSExNZsmQJr776KuvWraNVq1aAsdFg//79qVWrFm+++SaA6Z/Qs8TWpUsXxo4dS2RkJH5+fqb1u3fv5vr163Tt2hWALVu20K1bN5o0acKECRMAOHPmDHv27GHIkCFp7tvb25vp06fz9ttv06FDBzp27AhAlSpVTGWSk5MJCQmhfv36TJw40fT+HTBgAPPmzaNv3768++67hIWF8cMPP3D06FH27NmDra2tRe+PRYsWce/ePQYMGIBGo+Hrr7+mY8eOXLp0CVtbWwC2bt1KixYtKFmyJGPHjuXhw4dMnTqVevXqceTIkVS1BY8bPXo0n3/+OS1btqRly5YcOXKE5s2bk5iYmO42KRRFoW3btuzYsYPXX3+datWqsWnTJj744AOuXbvG5MmT8fb2ZuHChXzxxRfcv3+f8ePHA1C+fPk097l7925u377N0KFDM5Vc/P777zg4ONC5c+c0Hy9RogT169dn+/btPHz4EEdHR9NjTk5OtGvXjsWLF5suGR07doxTp07x888/c/z48aceHyApKYmrV69SuHBh07q///6bCxcuMHfuXOzs7OjYsSO//vor//vf/8y2LV68OPv27ePkyZNZuny/ZcsWWrdu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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "As can be seen from the above plots, our standard regression classifier successfully reduced the false-positive difference between African-Americans and Caucasions, **so on basis of this result we can says that our calssifier is more fair in comparison of COMPASS classsifier**!"
      ],
      "metadata": {
        "id": "C2gYehtbI2Oz"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Analyze performance on Overall Population :**\n",
        "\n",
        "To show the performance, ROC of overall population are as follows."
      ],
      "metadata": {
        "id": "gzZiFxIQy40E"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import roc_curve\n",
        "from matplotlib import pyplot as plt\n",
        "\n",
        "scores = logistic_regression.predict_proba(X_test)\n",
        "fpr, tpr, thresholds = roc_curve(y_test, scores[:,1])\n",
        "\n",
        "plt.plot(fpr, tpr)\n",
        "\n",
        "plt.title('ROC overall population')\n",
        "plt.xlabel(\"FPR\")\n",
        "plt.ylabel(\"TPR\")\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 472
        },
        "id": "1kn5UZ0ky2Cd",
        "outputId": "1212bdd0-e950-4a19-9a1b-fabd6582f6fc"
      },
      "execution_count": 137,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjcAAAHHCAYAAABDUnkqAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAzEUlEQVR4nO3de1xVZb7H8S8gewMqqKPclPFWal7SwiS8ZBaJl/T4mkzKjrcpNbNOSpa3ErVGzNT0pEk6qWU2mp50POrRinTMpHFCfU2NSpmaTgpKGhAWCDznj17s3AIKxN4blp/367VfM/vZz7PXb63K/fVZz1rLyxhjBAAAYBHeni4AAACgKhFuAACApRBuAACApRBuAACApRBuAACApRBuAACApRBuAACApRBuAACApRBuAACApRBuANRYd999t+6++27H+5MnT8rLy0urV6/2WE1VaeTIkWrWrFmVfufq1avl5eWlkydPVun3AtUJ4QZwkeIfkeJXrVq11LhxY40cOVLfffddqWOMMVqzZo3uuusu1atXTwEBAerQoYNmz56t3NzcMre1adMm9e3bVw0bNpTNZlN4eLiGDBmijz/+2FW7h2puzpw52rx5s6fLADyCcAO42OzZs7VmzRolJSWpb9++euedd9SzZ0/9/PPPTv0KCwv10EMPafjw4ZKkmTNnatGiRerUqZNmzZqlO++8UxkZGU5jjDEaNWqU/vCHPygjI0Px8fFKSkrS+PHjdfz4cd17773at2+f2/YV1UdZ4WbYsGH66aef1LRpU/cXBbhJLU8XAFhd37591blzZ0nSY489poYNG+rll1/Wli1bNGTIEEe/efPm6b333tOkSZP0yiuvONrHjBmjIUOGaNCgQRo5cqT+7//+z/HZggULtHr1ak2YMEELFy6Ul5eX47Pp06drzZo1qlWrZvxnbozRzz//LH9/f0+XYmk+Pj7y8fHxdBmASzFzA7hZjx49JEnffPONo+2nn37SK6+8olatWikxMbHEmAEDBmjEiBHasWOHPvvsM8eYxMREtWnTRvPnz3cKNsWGDRumLl26XLOe3NxcPfPMM4qIiJDdblfr1q01f/58GWMcfdq3b69evXqVGFtUVKTGjRtr8ODBTm2LFi1Su3bt5Ofnp5CQEI0dO1YXL150GtusWTPdf//92rlzpzp37ix/f3+98cYbkqRVq1bpnnvuUXBwsOx2u9q2batly5Zdcz8qoviU4Z49ezR27Fj97ne/U2BgoIYPH16iTkl6/fXX1a5dO9ntdoWHh2v8+PH64YcfnPrcfffdat++vVJTU9W1a1f5+/urefPmSkpKKnXbV6952b17t7y8vLR79+5r1j5//nx17dpVv/vd7+Tv76/IyEht3LjRqY+Xl5dyc3P11ltvOU6Ljhw58prbr8g+Hj58WL169VJAQIAaN26sefPmXbNmwN0IN4CbFf+o1K9f39G2d+9eXbx4UUOHDi1zpqX4dNXWrVsdYy5cuKChQ4dW+m/ixhgNHDhQr776qvr06aOFCxeqdevWevbZZxUfH+/oFxcXpz179ig9Pd1p/N69e3XmzBk99NBDjraxY8fq2WefVbdu3bR48WKNGjVKa9euVWxsrC5fvuw0Pi0tTQ8//LDuu+8+LV68WJ06dZIkLVu2TE2bNtW0adO0YMECRURE6IknntDSpUsrtZ9lefLJJ3XkyBHNnDlTw4cP19q1azVo0CCnYDdz5kyNHz9e4eHhWrBggR544AG98cYb6t27d4n9uXjxovr166fIyEjNmzdPTZo00bhx47Ry5coqq3nx4sW67bbbNHv2bM2ZM0e1atXSgw8+qG3btjn6rFmzRna7XT169NCaNWu0Zs0ajR07tszvrOg+9unTRx07dtSCBQvUpk0bTZ482WlGEfA4A8AlVq1aZSSZjz76yJw/f96cPn3abNy40TRq1MjY7XZz+vRpR99FixYZSWbTpk1lft+FCxeMJPOHP/zBGGPM4sWLrzvmejZv3mwkmZdeesmpffDgwcbLy8scO3bMGGNMWlqakWRee+01p35PPPGEqVOnjrl06ZIxxphPPvnESDJr16516rdjx44S7U2bNjWSzI4dO0rUVfx9V4qNjTUtWrRwauvZs6fp2bOn4/2JEyeMJLNq1apr7nfxP5vIyEiTn5/vaJ83b56RZP76178aY4w5d+6csdlspnfv3qawsNDRb8mSJUaSWblypVMtksyCBQscbXl5eaZTp04mODjYsZ3ibZ84ccKppl27dhlJZteuXY62ESNGmKZNm17z2OTn55v27dube+65x6m9du3aZsSIEWXue/H2K7OPb7/9ttM+hoaGmgceeKDEtgBPYeYGcLGYmBg1atRIERERGjx4sGrXrq0tW7aoSZMmjj45OTmSpLp165b5PcWfZWdnO/3vtcZcz/bt2+Xj46P/+q//cmp/5plnZIxx/G28VatW6tSpk9avX+/oU1hYqI0bN2rAgAGOdTIbNmxQUFCQ7rvvPmVmZjpekZGRqlOnjnbt2uW0nebNmys2NrZEXVeuu8nKylJmZqZ69uyp48ePKysrq9L7e7UxY8bI19fX8X7cuHGqVauWtm/fLkn66KOPlJ+frwkTJsjb+9c/LkePHq3AwECn2RJJqlWrltMMic1m09ixY3Xu3DmlpqZWSc1XHpuLFy8qKytLPXr00IEDByr1fRXdxzp16ug///M/He9tNpu6dOmi48ePV2r7gCvUjJWGQA22dOlStWrVSllZWVq5cqX27Nkju93u1Kc4oBSHnNJcHYACAwOvO+Z6vv32W4WHh5cISLfccovj82JxcXGaNm2avvvuOzVu3Fi7d+/WuXPnFBcX5+jz9ddfKysrS8HBwaVu79y5c07vmzdvXmq/Tz/9VAkJCUpJSdGlS5ecPsvKylJQUFD5d/Iabr75Zqf3derUUVhYmOPUYfH+t27d2qmfzWZTixYtnI6PJIWHh6t27dpOba1atZL0y+nIO++88zfXvHXrVr300ks6dOiQ8vLyHO2lrbkqj4ruY5MmTUpsq379+vrnP/9Zqe0DrkC4AVysS5cujqulBg0apO7du2vo0KFKS0tTnTp1JP0aJv75z39q0KBBpX5P8Y9H27ZtJUlt2rSRJH3xxRdljqlKcXFxmjp1qjZs2KAJEybovffeU1BQkPr06ePoU1RUpODgYK1du7bU72jUqJHT+9KujPrmm2907733qk2bNlq4cKEiIiJks9m0fft2vfrqqyoqKqraHXOzskJIYWHhdcd+8sknGjhwoO666y69/vrrCgsLk6+vr1atWqV33323qkstVVnru8wV65QAT+O0FOBGPj4+SkxM1JkzZ7RkyRJHe/fu3VWvXj29++67Zf7Ivf3225Kk+++/3zGmfv36+stf/lKuH8bSNG3aVGfOnCkx+3P06FHH58WaN2+uLl26aP369SooKND777+vQYMGOc1CtWzZUt9//726deummJiYEq+OHTtet6b//d//VV5enrZs2aKxY8eqX79+iomJcckl4l9//bXT+x9//FFnz5513BW4eP/T0tKc+uXn5+vEiRMl7hVz5syZEjdb/OqrryTJ8Z3FC8mvvhLp6hmS0vzP//yP/Pz8tHPnTv3xj39U3759FRMTU2rf8s7kVHQfgZqAcAO42d13360uXbpo0aJFjhv5BQQEaNKkSUpLS9P06dNLjNm2bZtWr16t2NhYx6mNgIAATZ48WUeOHNHkyZNL/ZvzO++8o/3795dZS79+/VRYWOgUtCTp1VdflZeXl/r27evUHhcXp88++0wrV65UZmam0ykpSRoyZIgKCwv14osvlthWQUFBiR/00hTPDFy5P1lZWVq1atV1x1bU8uXLna4GWrZsmQoKChz7HRMTI5vNpv/+7/92qufNN99UVlaW+vfv7/R9BQUFjsvZpV8CwhtvvKFGjRopMjJS0i8BUJL27Nnj6FdYWKjly5dft14fHx95eXk5hdmTJ0+WerO+2rVrl+t4V3QfgZqA01KABzz77LN68MEHtXr1aj3++OOSpClTpujgwYN6+eWXlZKSogceeED+/v7au3ev3nnnHd1yyy166623SnzPv/71Ly1YsEC7du3S4MGDFRoaqvT0dG3evFn79++/5h2KBwwYoF69emn69Ok6efKkOnbsqA8++EB//etfNWHCBMcPcbEhQ4Zo0qRJmjRpkho0aFBi1qBnz54aO3asEhMTdejQIfXu3Vu+vr76+uuvtWHDBi1evNjpnjil6d27t2w2mwYMGKCxY8fqxx9/1IoVKxQcHKyzZ89W5DBfV35+vu69914NGTJEaWlpev3119W9e3cNHDhQ0i+n0aZOnapZs2apT58+GjhwoKPfHXfc4bSwVvplzc3LL7+skydPqlWrVlq/fr0OHTqk5cuXOxYut2vXTnfeeaemTp2qCxcuqEGDBlq3bp0KCgquW2///v21cOFC9enTR0OHDtW5c+e0dOlS3XTTTSXWvERGRuqjjz7SwoULFR4erubNmysqKqrEd1Z0H4EawZOXagFWVnzJ7T/+8Y8SnxUWFpqWLVuali1bmoKCAqf2VatWmW7dupnAwEDj5+dn2rVrZ2bNmmV+/PHHMre1ceNG07t3b9OgQQNTq1YtExYWZuLi4szu3buvW2dOTo6ZOHGiCQ8PN76+vubmm282r7zyiikqKiq1f7du3Ywk89hjj5X5ncuXLzeRkZHG39/f1K1b13To0ME899xz5syZM44+TZs2Nf379y91/JYtW8ytt95q/Pz8TLNmzczLL79sVq5cWeIS6t96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cZyvN2NTjAXFAADAUgg3AADAUgg3VchutyshIUF2u93TpVgax9k9OM7uwXF2H461e1SH43zDLSgGAADWxswNAACwFMINAACwFMINAACwFMINAACwFMJNBS1dulTNmjWTn5+foqKitH///mv237Bhg9q0aSM/Pz916NBB27dvd1OlNVtFjvOKFSvUo0cP1a9fX/Xr11dMTMx1/7ngFxX997nYunXr5OXlpUGDBrm2QIuo6HH+4YcfNH78eIWFhclut6tVq1b82VEOFT3OixYtUuvWreXv76+IiAhNnDhRP//8s5uqrZn27NmjAQMGKDw8XF5eXtq8efN1x+zevVu333677Ha7brrpJq1evdrldcqg3NatW2dsNptZuXKl+de//mVGjx5t6tWrZzIyMkrt/+mnnxofHx8zb948c/jwYfP8888bX19f88UXX7i58pqlosd56NChZunSpebgwYPmyJEjZuTIkSYoKMj8+9//dnPlNUtFj3OxEydOmMaNG5sePXqY//iP/3BPsTVYRY9zXl6e6dy5s+nXr5/Zu3evOXHihNm9e7c5dOiQmyuvWSp6nNeuXWvsdrtZu3atOXHihNm5c6cJCwszEydOdHPlNcv27dvN9OnTzfvvv28kmU2bNl2z//Hjx01AQICJj483hw8fNq+99prx8fExO3bscGmdhJsK6NKlixk/frzjfWFhoQkPDzeJiYml9h8yZIjp37+/U1tUVJQZO3asS+us6Sp6nK9WUFBg6tata9566y1XlWgJlTnOBQUFpmvXrubPf/6zGTFiBOGmHCp6nJctW2ZatGhh8vPz3VWiJVT0OI8fP97cc889Tm3x8fGmW7duLq3TSsoTbp577jnTrl07p7a4uDgTGxvrwsqM4bRUOeXn5ys1NVUxMTGONm9vb8XExCglJaXUMSkpKU79JSk2NrbM/qjccb7apUuXdPnyZTVo0MBVZdZ4lT3Os2fPVnBwsB599FF3lFnjVeY4b9myRdHR0Ro/frxCQkLUvn17zZkzR4WFhe4qu8apzHHu2rWrUlNTHaeujh8/ru3bt6tfv35uqflG4anfwRvuwZmVlZmZqcLCQoWEhDi1h4SE6OjRo6WOSU9PL7V/enq6y+qs6SpznK82efJkhYeHl/gPCr+qzHHeu3ev3nzzTR06dMgNFVpDZY7z8ePH9fHHH+uRRx7R9u3bdezYMT3xxBO6fPmyEhIS3FF2jVOZ4zx06FBlZmaqe/fuMsaooKBAjz/+uKZNm+aOkm8YZf0OZmdn66effpK/v79LtsvMDSxl7ty5WrdunTZt2iQ/Pz9Pl2MZOTk5GjZsmFasWKGGDRt6uhxLKyoqUnBwsJYvX67IyEjFxcVp+vTpSkpK8nRplrJ7927NmTNHr7/+ug4cOKD3339f27Zt04svvujp0lAFmLkpp4YNG8rHx0cZGRlO7RkZGQoNDS11TGhoaIX6o3LHudj8+fM1d+5cffTRR7r11ltdWWaNV9Hj/M033+jkyZMaMGCAo62oqEiSVKtWLaWlpally5auLboGqsy/z2FhYfL19ZWPj4+j7ZZbblF6erry8/Nls9lcWnNNVJnj/MILL2jYsGF67LHHJEkdOnRQbm6uxowZo+nTp8vbm7/7V4WyfgcDAwNdNmsjMXNTbjabTZGRkUpOTna0FRUVKTk5WdHR0aWOiY6OduovSR9++GGZ/VG54yxJ8+bN04svvqgdO3aoc+fO7ii1RqvocW7Tpo2++OILHTp0yPEaOHCgevXqpUOHDikiIsKd5dcYlfn3uVu3bjp27JgjPErSV199pbCwMIJNGSpznC9dulQiwBQHSsMjF6uMx34HXbpc2WLWrVtn7Ha7Wb16tTl8+LAZM2aMqVevnklPTzfGGDNs2DAzZcoUR/9PP/3U1KpVy8yfP98cOXLEJCQkcCl4OVT0OM+dO9fYbDazceNGc/bsWccrJyfHU7tQI1T0OF+Nq6XKp6LH+dSpU6Zu3brmySefNGlpaWbr1q0mODjYvPTSS57ahRqhosc5ISHB1K1b1/zlL38xx48fNx988IFp2bKlGTJkiKd2oUbIyckxBw8eNAcPHjSSzMKFC83BgwfNt99+a4wxZsqUKWbYsGGO/sWXgj/77LPmyJEjZunSpVwKXh299tpr5ve//72x2WymS5cu5rPPPnN81rNnTzNixAin/u+9955p1aqVsdlspl27dmbbtm1urrhmqshxbtq0qZFU4pWQkOD+wmuYiv77fCXCTflV9Djv27fPREVFGbvdblq0aGH+9Kc/mYKCAjdXXfNU5DhfvnzZzJw507Rs2dL4+fmZiIgI88QTT5iLFy+6v/AaZNeuXaX+eVt8bEeMGGF69uxZYkynTp2MzWYzLVq0MKtWrXJ5nV7GMP8GAACsgzU3AADAUgg3AADAUgg3AADAUgg3AADAUgg3AADAUgg3AADAUgg3AADAUgg3AADAUgg3AKq9kSNHysvLq8Tr2LFjTp/ZbDbddNNNmj17tgoKCiT98vTnK8c0atRI/fr10xdffOHhvQLgKoQbADVCnz59dPbsWadX8+bNnT77+uuv9cwzz2jmzJl65ZVXnManpaXp7Nmz2rlzp/Ly8tS/f3/l5+d7YlcAuBjhBkCNYLfbFRoa6vQqfopz8WdNmzbVuHHjFBMToy1btjiNDw4OVmhoqG6//XZNmDBBp0+f1tGjRz2xKwBcjHADwHL8/f3LnJXJysrSunXrJEk2m82dZQFwk1qeLgAAymPr1q2qU6eO433fvn21YcMGpz7GGCUnJ2vnzp166qmnnD5r0qSJJCk3N1eSNHDgQLVp08bFVQPwBMINgBqhV69eWrZsmeN97dq1Hf+/OPhcvnxZRUVFGjp0qGbOnOk0/pNPPlFAQIA+++wzzZkzR0lJSe4qHYCbEW4A1Ai1a9fWTTfdVOpnxcHHZrMpPDxctWqV/KOtefPmqlevnlq3bq1z584pLi5Oe/bscXXZADyANTcAarzi4PP73/++1GBztfHjx+vLL7/Upk2b3FAdAHcj3AC44QQEBGj06NFKSEiQMcbT5QCoYoQbADekJ598UkeOHCmxKBlAzedl+GsLAACwEGZuAACApRBuAACApRBuAACApRBuAACApRBuAACApRBuAACApRBuAACApRBuAACApRBuAACApRBuAACApRBuAACApRBuAACApfw/kTcm3nT8j9kAAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "## Part 3: Fair Classifiers\n",
        "\n",
        "There are various techniques to build a fair model.\n",
        "\n",
        "#### 1) USING FAIR LIBRARY (1st Experiment)\n",
        "##### AI Fairness 360 (AIF360) :\n",
        "\n",
        "AI Fairness 360 (AIF360) library address issues of fairness, transparency, and accountability in artificial intelligence (AI) and machine learning (ML) systems. Specifically, AIF360 aims to:\n",
        "1.  Provide a comprehensive set of fairness metrics\n",
        "2.  Develop bias mitigation algorithms\n",
        "3.  Support both structured and unstructured data\n",
        "4.  Promote transparency and reproducibility\n",
        "**Cons**: May have a steeper learning curve due to its wide range of functionalities.\n",
        "\n",
        "Beside this, We also have a Fairlearn library that helps to make sure that machine learning models treat everyone fairly. It offers tools to check if models are biased and techniques to fix any unfairness. Fairlearn works well with a popular machine learning library called scikit-learn, making it easy for people to use. It also provides easy-to-understand graphs and visuals to help people see if their models are fair or not. Overall, Fairlearn aims to help developers and data scientists create fairer machine learning models that treat everyone equally.\n",
        "\n",
        "Now, we build a fair classifier. First, we describe the different methods to obtain a fair classifier. Then, we present our implementation."
      ],
      "metadata": {
        "id": "b9HoQbCDZkbY"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install aif360\n",
        "from aif360.datasets import BinaryLabelDataset\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.metrics import accuracy_score\n",
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "# Assuming 'data' contains all features including sensitive attributes and the target variable\n",
        "privileged_groups = [{'race_Caucasian': 1}]  # Assuming 'race' is a protected attribute\n",
        "unprivileged_groups = [{'race_African-American': 1}]\n",
        "favorable_label = 1  # Assuming '1' indicates the favorable outcome in the target variable\n",
        "unfavorable_label = 0  # Assuming '0' indicates the unfavorable outcome in the target variable\n",
        "\n",
        "FEATURES_CLASSIFICATION = [\"race_African-American\", \"race_Asian\", \"race_Caucasian\", \"race_Hispanic\", \"race_Native American\", \"race_Other\",\n",
        "                           \"age\", \"sex_Female\", \"sex_Male\", \"priors_count\", \"c_charge_degree_F\", \"c_charge_degree_M\"] #features to be used for classification\n",
        "CONT_VARIABLES = [\"priors_count\"] # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot\n",
        "CLASS_FEATURE = \"two_year_recid\" # the decision variable\n",
        "SENSITIVE_ATTRS = [\"race_African-American\", \"race_Asian\", \"race_Caucasian\", \"race_Hispanic\", \"race_Native American\", \"race_Other\"]\n",
        "\n",
        "'''FEATURES_CLASSIFICATION = [\"age_cat\", \"race\", \"sex\", \"priors_count\", \"c_charge_degree\"] #features to be used for classification\n",
        "CONT_VARIABLES = [\"priors_count\"] # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot\n",
        "CLASS_FEATURE = \"two_year_recid\" # the decision variable\n",
        "SENSITIVE_ATTRS = [\"race\"]'''\n",
        "\n",
        "fair_dataset = pd.concat([X, y], axis=1)\n",
        "\n",
        "# Create a BinaryLabelDataset object\n",
        "dataset = BinaryLabelDataset(\n",
        "    favorable_label=favorable_label,\n",
        "    unfavorable_label=unfavorable_label,\n",
        "    df=fair_dataset,  # Assuming 'data' contains all features including sensitive attributes and the target variable\n",
        "    label_names=[CLASS_FEATURE],  # Name of the target variable\n",
        "    protected_attribute_names=[\"race_African-American\", \"race_Asian\", \"race_Caucasian\", \"race_Hispanic\", \"race_Native American\", \"race_Other\"],  # List of sensitive attribute names\n",
        "    unprivileged_protected_attributes=unprivileged_groups\n",
        ")\n",
        "\n",
        "# Split the dataset into train and test sets\n",
        "train, test = dataset.split([0.7], shuffle=True, seed=1234)\n",
        "\n",
        "# Train a logistic regression model\n",
        "model = LogisticRegression()\n",
        "model.fit(train.features, train.labels.ravel())\n",
        "\n",
        "# Predict on the test set\n",
        "test_pred = model.predict(test.features)\n",
        "accuracy = accuracy_score(test.labels.ravel(), test_pred)\n",
        "print(\"Accuracy:\", accuracy)\n",
        "\n",
        "# Assess bias\n",
        "from aif360.metrics import BinaryLabelDatasetMetric\n",
        "metric = BinaryLabelDatasetMetric(test, privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)\n",
        "print(\"Disparate impact:\", metric.disparate_impact())"
      ],
      "metadata": {
        "id": "rYypbAKkZnhs",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "585109be-437f-4e97-cef1-5c448d34937c"
      },
      "execution_count": 138,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: aif360 in /usr/local/lib/python3.10/dist-packages (0.6.1)\n",
            "Requirement already satisfied: numpy>=1.16 in /usr/local/lib/python3.10/dist-packages (from aif360) (1.25.2)\n",
            "Requirement already satisfied: scipy>=1.2.0 in /usr/local/lib/python3.10/dist-packages (from aif360) (1.11.4)\n",
            "Requirement already satisfied: pandas>=0.24.0 in /usr/local/lib/python3.10/dist-packages (from aif360) (2.0.3)\n",
            "Requirement already satisfied: scikit-learn>=1.0 in /usr/local/lib/python3.10/dist-packages (from aif360) (1.2.2)\n",
            "Requirement already satisfied: matplotlib in /usr/local/lib/python3.10/dist-packages (from aif360) (3.7.1)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas>=0.24.0->aif360) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=0.24.0->aif360) (2023.4)\n",
            "Requirement already satisfied: tzdata>=2022.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=0.24.0->aif360) (2024.1)\n",
            "Requirement already satisfied: joblib>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from scikit-learn>=1.0->aif360) (1.4.0)\n",
            "Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn>=1.0->aif360) (3.4.0)\n",
            "Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->aif360) (1.2.1)\n",
            "Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib->aif360) (0.12.1)\n",
            "Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->aif360) (4.51.0)\n",
            "Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->aif360) (1.4.5)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->aif360) (24.0)\n",
            "Requirement already satisfied: pillow>=6.2.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->aif360) (9.4.0)\n",
            "Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->aif360) (3.1.2)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas>=0.24.0->aif360) (1.16.0)\n",
            "Accuracy: 0.5326850351161534\n",
            "Disparate impact: 1.324496379845343\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#### 2) Without using any FAIR LIBRARY (2nd Experiment)\n",
        "\n",
        "**Can we build a fair classifier by simply excluding the races from the input?**"
      ],
      "metadata": {
        "id": "G4V9dvoy1aAf"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "No, we don't think that a fair classifier can be implemented simply by excluding the sensitive attributes.\n",
        "Because if input data is biased, for example, if we have a subgroup that is minority represented inside our training data, then when we optimize the loss function, it's possible the best loss function will be as accurate as possible on the majority group while incurring errors on the minority group.\n",
        "So we propose sampling data uniformly from each racial group with same size before applying logistic regression on it as the method to implement fairness classifier.\n"
      ],
      "metadata": {
        "id": "ya3hwvgz5KNe"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import random\n",
        "race_datasets = {}\n",
        "fairdataset = pd.DataFrame()\n",
        "samplesize = 2000\n",
        "random.seed(42)\n",
        "for race in dataset_fair_model['race'].unique():\n",
        "    race_datasets[race] = dataset_fair_model[dataset_fair_model['race'] == race]\n",
        "    if (len(race_datasets[race]) >= samplesize) :\n",
        "        #random_state should use a random seed\n",
        "        sampledata = race_datasets[race].sample(n=samplesize, random_state=random.randint(1,100))\n",
        "        fairdataset = pd.concat([fairdataset, sampledata])\n",
        "\n",
        "print(\"Shape of fair dataset after is: \", fairdataset.shape)\n",
        "\n",
        "# Select the features that we find relevant from the dataset\n",
        "fairdataset = fairdataset[[\"sex\", \"dob\", \"age\", \"age_cat\", \"race\", \"juv_fel_count\", \"juv_misd_count\", \"juv_other_count\",\n",
        "                  \"priors_count\", \"c_charge_degree\", \"start\", \"end\", \"event\",\n",
        "                  \"decile_score\", \"score_text\", \"v_decile_score\", \"two_year_recid\", \"is_violent_recid\"]]\n",
        "\n",
        "# Now, to show that missing values have been removed, we calculate and print the sum of missing values in every column\n",
        "missing_values = fairdataset.isnull().sum()\n",
        "print(\"Below is the number of missing values in every column:\")\n",
        "print(missing_values)\n",
        "\n",
        "# Select the dependent variable\n",
        "y = fairdataset[[\"two_year_recid\"]]\n",
        "\n",
        "# Select the independent variables\n",
        "X = fairdataset[[\"sex\", \"dob\", \"age\", \"juv_fel_count\", \"juv_misd_count\", \"juv_other_count\", \"priors_count\",\n",
        "             \"c_charge_degree\", \"start\", \"end\", \"event\", \"race\"]]\n",
        "\n",
        "# Convert \"sex\", \"race\", and \"c_charge_degree\" to numerical data (dummies)\n",
        "X = pd.get_dummies(X, columns=[\"sex\", \"c_charge_degree\", \"race\"], dtype=int)\n",
        "\n",
        "# Convert \"dob\" to integer\n",
        "X[\"dob\"] = pd.to_datetime(X[\"dob\"]).astype(np.int64)\n",
        "\n",
        "# Print the first few rows of X to confirm that all non-numerical data have correctly been converted to numerical data\n",
        "X.head()\n",
        "\n",
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y)\n",
        "\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.pipeline import Pipeline\n",
        "\n",
        "logistic_regression = Pipeline([(\"scaler\", StandardScaler()), (\"classifier\", LogisticRegression())])\n",
        "logistic_regression.fit(X_train, y_train[\"two_year_recid\"])\n",
        "predictions = logistic_regression.predict_proba(X_test)\n",
        "\n",
        "from sklearn.metrics import roc_curve\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "def plot_false_positive_rate_vs_threshold_of_fair_classifier(races, predicted_probabilities):\n",
        "    for current in races:\n",
        "        current_race = (X_test[\"race\" + \"_\" + current] == 1)\n",
        "        ground_truth = y_test[current_race][\"two_year_recid\"]\n",
        "        predictions = predicted_probabilities[current_race][:,1]\n",
        "        FPR, TPR, thresholds = roc_curve(ground_truth, predictions)\n",
        "        plt.plot(thresholds, FPR, label=current)\n",
        "    plt.ylabel(\"False Positive Rate\")\n",
        "    plt.xlabel(\"Threshold\")\n",
        "    plt.title(\"False positive rate vs treshold of fair classifier\")\n",
        "    plt.legend()\n",
        "    plt.show()\n",
        "\n",
        "races = np.array([\"African-American\", \"Caucasian\"])\n",
        "plot_false_positive_rate_vs_threshold_of_fair_classifier(races, predictions)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 837
        },
        "id": "u3hHVrre_YNw",
        "outputId": "91e34961-6a92-44ff-f7d1-d969291dd6f6"
      },
      "execution_count": 139,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Shape of fair dataset after is:  (4000, 53)\n",
            "Below is the number of missing values in every column:\n",
            "sex                 0\n",
            "dob                 0\n",
            "age                 0\n",
            "age_cat             0\n",
            "race                0\n",
            "juv_fel_count       0\n",
            "juv_misd_count      0\n",
            "juv_other_count     0\n",
            "priors_count        0\n",
            "c_charge_degree     0\n",
            "start               0\n",
            "end                 0\n",
            "event               0\n",
            "decile_score        0\n",
            "score_text          0\n",
            "v_decile_score      0\n",
            "two_year_recid      0\n",
            "is_violent_recid    0\n",
            "dtype: int64\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Result :**\n",
        "\n",
        "From the result we could observe that this crude fair classifier is much fairer than the COMPAS classifier because the false positive rate difference between African-American and Caucasian is much smaller. And we even achieve minor improvement on false positive rate between African-American and Caucasian than the standard logistic regression in the low threshold area.\n"
      ],
      "metadata": {
        "id": "1BzCyVmMbIv7"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Conclusion Note:**\n",
        "\n",
        "After analysing our problem and data. We have studied a lot of research articals and libraries related to the fair model and we end up the the crude fair classifier. In Conclusion, our crude fair model false positive rate difference between African-American and Caucasian is smaller. So, Our model is fair in comparison of COMPAS.\n"
      ],
      "metadata": {
        "id": "nqeh2H3c4KOm"
      }
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "YzUWIhzE6Om5"
      },
      "execution_count": null,
      "outputs": []
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3 (ipykernel)",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.10.12"
    },
    "toc": {
      "base_numbering": 1,
      "nav_menu": {},
      "number_sections": false,
      "sideBar": false,
      "skip_h1_title": false,
      "title_cell": "Table of Contents",
      "title_sidebar": "Contents",
      "toc_cell": false,
      "toc_position": {},
      "toc_section_display": false,
      "toc_window_display": false
    },
    "colab": {
      "provenance": []
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
