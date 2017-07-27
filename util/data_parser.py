#!/usr/bin/env python
"""
This module will provide the functionalities for parsing the data.
"""
import mysql.connector as MySQLdb
import json
import os
import csv


class DataParser(object):
    """
    A class for parsing given data files.
    """

    def __init__(self, n_users=None):
        self.n_users = n_users

    def get_ratings_hash(self):
        """
        :returns: A dictionary of user_id to a list of paper_id, of the papers this user rated.
        :rtype: dict
        """
        db = DataParser.get_connection()
        cursor = db.cursor()
        config = DataParser.get_config()
        cursor.execute("use %s" % config["database"]["database_name"])
        cursor.execute("set group_concat_max_len=100000")
        if self.n_users is None:
            cursor.execute("select user_id, group_concat(article_id separator ', ') from articles_users" +
                           " group by user_id")
        else:
            cursor.execute("select user_id, group_concat(article_id separator ', ') from articles_users" +
                           " group by user_id order by user_id limit %s" % self.n_users)
        ratings_hash = {}
        all_papers = set()
        for (user_id, json_object) in cursor:
            papers = DataParser.listify(json_object)
            ratings_hash[int(user_id) - 1] = papers
            if self.n_users is not None:
                all_papers = all_papers.union(set(papers))
        DataParser.clean_up(db, cursor)
        if self.n_users is not None:
            self.papers = list(sorted(all_papers))
            self.papers_map = {}
            for i, paper in enumerate(self.papers):
                self.papers_map[paper] = i
        return ratings_hash

    def get_ratings_matrix(self):
        """
        :returns:
            Matrix between users and documents. 1 indicates that the user has the document
            in his library, 0 otherwise.
        :rtype: int[][]
        """
        ratings_hash = self.get_ratings_hash()
        num_users = DataParser.get_row_count("users")
        num_articles = DataParser.get_row_count("articles")
        if self.n_users is not None:
            num_users = self.n_users
            num_articles = len(self.papers)
        ratings_matrix = [[0] * num_articles for _ in range(num_users)]
        for user_id, articles in ratings_hash.items():
            for article_id in articles:
                #Bug discovered by Anas on 19.7, old code:
                #ratings_matrix[user_id - 1][article_id - 1] = 1
                #Correct code:
                if self.n_users is None:
                    ratings_matrix[user_id][article_id - 1] = 1
                else:
                    ratings_matrix[user_id][self.papers_map[article_id]] = 1
        return ratings_matrix

    def get_word_distribution(self):
        """
        The function return metrics of word distributions with articles. First is a list of articles and words.
        Second is a list of words and their counts, third is a list of words, articles and their counts.

        :returns: a triple of arrays
        :rtype: triple
        """
        db = DataParser.get_connection()
        cursor = db.cursor()
        config = DataParser.get_config()
        cursor.execute("use %s" % config["database"]["database_name"])
        if self.n_users is None:
            cursor.execute("select article_id, word_id from words_articles order by article_id, word_id")
            article_words = cursor.fetchall()
            article_words = list(map(lambda t: (t[0] - 1, t[1] - 1), article_words))
            cursor.execute("select article_id, word_id, count as word_count"
                           " from words_articles order by article_id, word_id")
            article_word_count = cursor.fetchall()
            article_word_count = list(map(lambda t: (t[0] - 1, t[1] - 1, t[2]), article_word_count))
            cursor.execute("select word_id, count(*) as word_count from" + 
                           " words_articles group by word_id order by word_id")
            word_count = cursor.fetchall()
            word_count = list(map(lambda t: (t[0] - 1, t[1]), word_count))
        else:
            query_string = ', '.join(list(map(lambda x: '%s', self.papers)))
            cursor.execute("select article_id, word_id from words_articles where article_id in" +  
                           " (%s) order by article_id, word_id" % query_string, self.papers)
            article_words = cursor.fetchall()
            article_words = list(map(lambda t: (self.papers_map[t[0]], t[1] - 1), article_words))
            cursor.execute("select article_id, word_id, count as word_count"
                           " from words_articles where article_id in (%s) order by article_id, word_id" % query_string, self.papers)
            article_word_count = cursor.fetchall()
            article_word_count = list(map(lambda t: (self.papers_map[t[0]], t[1] - 1, t[2]), article_word_count))
            cursor.execute("select word_id, count(*) as word_count from words_articles" +
                           " where article_id in (%s) group by word_id order by word_id" % query_string, self.papers)
            word_count = cursor.fetchall()
            word_count = list(map(lambda t: (t[0] - 1, t[1]), word_count))

        return word_count, article_words, article_word_count

    @staticmethod
    def process():
        """
        Start processing the data. before running make sure you set autocommit to 0 in mysql,
        otherwise the script might take too long to execute
        """
        db = DataParser.get_connection()
        cursor = db.cursor()
        DataParser.set_up_database(cursor)
        config = DataParser.get_config()
        cursor.execute("use %s" % config["database"]["database_name"])
        DataParser.import_articles(cursor)
        DataParser.import_citations(cursor)
        DataParser.import_words(cursor)
        DataParser.import_users(cursor)
        DataParser.clean_up(db, cursor)

    @staticmethod
    def listify(input_str):
        """
        Transfer a string of comma separated integers into a list of integers.

        :param str input_str: comma separated ints
        :returns: representation of the input
        :rtype: list[int]
        """
        return list(map(int, input_str.split(',')))

    @staticmethod
    def get_row_count(table_name):
        """
        :returns: indicating number of users
        :rtype: int
        """
        db = DataParser.get_connection()
        cursor = db.cursor()
        config = DataParser.get_config()
        cursor.execute("use %s" % config["database"]["database_name"])
        cursor.execute("select count(*) as c from %s" % table_name)
        row = cursor.fetchone()
        return int(row[0])


    @staticmethod
    def import_articles(cursor):
        """
        reads raw-data.csv and fills the articles table
        """
        print("*** Inserting Articles ***")
        dataset = DataParser.get_dataset()
        first_line = True
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "data",
                  dataset, "raw-data.csv"), "r", encoding='utf-8', errors='ignore') as f:
            delimiter = '\t'
            if dataset == 'citeulike-t':
                reader = csv.reader(f, quotechar='"', delimiter=delimiter)
            elif dataset == 'citeulike-a':
                reader = csv.reader(f, quotechar='"')
            for line in reader:
                if first_line:
                    first_line = False
                    continue
                if dataset == 'citeulike-t':
                    id = int(line[0]) + 1
                    title = ""
                elif dataset == 'citeulike-a':
                    id = int(line[0])
                    title = line[1]

                if DataParser.store_abstracts():
                    if dataset == 'citeulike-t':
                        abstract = line[1]
                    else:
                        abstract = line[4]
                else:
                    abstract = ""
                cursor.execute("insert into articles(id, title, abstract) values(%s, \"%s\", \"%s\")",
                               (str(id), title, abstract.replace("\"", "\\\"")))

    @staticmethod
    def import_citations(cursor):
        """
        reads citations.dat and inserts rows in the citations table
        """
        print("*** Inserting Citations ***")
        id = 1
        dataset = DataParser.get_dataset()
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "data",
                  dataset, "citations.dat")) as f:
            for line in f:
                splitted = line.replace("\n", "").split(" ")
                num_citations = splitted[0]
                for i in range(1, int(num_citations) + 1):
                    cursor.execute("insert into citations(article_id, cited_article_id) \
                                   values (%s,%s)", (id, splitted[i]))
                id += 1

    @staticmethod
    def import_words(cursor):
        """
        reads mult.dat and vocabulary.dat to insert bag of words representation in words_articles
        """
        print("*** Inserting Words ***")
        base_dir = os.path.dirname(os.path.realpath(__file__))
        id = 1
        dataset = DataParser.get_dataset()
        with open(os.path.join(os.path.dirname(base_dir), "data", dataset, "mult.dat")) as\
                bag, open(os.path.join(os.path.dirname(base_dir), "data", dataset, "vocabulary.dat")) as vocab:
            for entry in bag:
                entry = entry.strip()
                splitted = entry.split(" ")
                num_words = int(splitted[0])
                for i in range(1, num_words + 1):
                    article_to_count = splitted[i].split(":")
                    word_id = str(int(article_to_count[0]) + 1)
                    count = article_to_count[1]
                    cursor.execute("insert into words_articles(article_id, count, word_id) \
                                   values (%s, %s, %s)", (id, count, word_id))
                id += 1
            current_word = 1
            for word in vocab:
                word = word.strip()
                cursor.execute("insert ignore into words(id, word) values(%s, %s)", (current_word, word))
                current_word += 1

    @staticmethod
    def import_users(cursor):
        """
        reads users.dat to insert entries in users and articles_users table
        """
        print("*** Inserting Users ***")
        id = 1
        dataset = DataParser.get_dataset()
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "data",
                  dataset, "users.dat")) as f:
            for line in f:
                splitted = line.replace("\n", "").split(" ")
                num_articles = int(splitted[0])

                cursor.execute("insert into users(id) values(%s)" % id)
                for i in range(1, num_articles + 1):
                    if dataset == 'citeulike-t':
                        article_id = int(splitted[i])
                    elif dataset == 'citeulike-a':
                        article_id = int(splitted[i]) + 1
                    cursor.execute("insert into articles_users(user_id, article_id) values(%s, %s)", (id, article_id))
                id += 1

    @staticmethod
    def get_config():
        """
        :returns: representation of the config file
        :rtype: dict
        """
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                               'config/config.json')) as data_file:
            data = json.load(data_file)
        return data

    @staticmethod
    def get_dataset():
        """
        :returns: The dataset file name
        :rtype: str
        """
        dataset = DataParser.get_config()["dataset"]
        if dataset != 'citeulike-a' and dataset != 'citeulike-t':
            raise NameError("'citeulike-a' and 'citeulike-t' are the only valid datasets")
        return dataset

    @staticmethod
    def store_abstracts():
        """
        :returns: A flag deciding storing abstracts.
        :rtype: bool
        """
        return DataParser.get_config()["store_abstracts"]

    @staticmethod
    def get_abstracts():
        """
        :returns: the key is document id, value is the document's abstract
        :rtype: dict
        """
        db = DataParser.get_connection()
        cursor = db.cursor()
        config = DataParser.get_config()
        cursor.execute("use %s" % config["database"]["database_name"])
        cursor.execute("select id, replace(abstract, \"'\", \"\") as abstract from articles")
        abstracts = dict()
        for id, abstract in cursor:
            abstracts[id - 1] = abstract
        return abstracts

    @staticmethod
    def get_connection():
        """
        :returns: A database connection.
        """
        config = DataParser.get_config()
        db = MySQLdb.connect(host=config["database"]["host"], user=config["database"]["user"],
                             passwd=config["database"]["password"])
        return db

    @staticmethod
    def set_up_database(cursor):
        """
        Creates the mysql tables in the database
        """
        config = DataParser.get_config()
        cursor.execute("create database if not exists %s" % (config["database"]["database_name"]))
        cursor.execute("use %s" % config["database"]["database_name"])
        cursor.execute("create table if not exists users(id int(11) not null auto_increment, primary key(id))")
        cursor.execute("create table if not exists articles(id int(11) not null auto_increment, " +
                       "abstract text character set utf8mb4 COLLATE utf8mb4_general_ci not null,\
                       title varchar(255) not null, primary key(id))")
        cursor.execute("create table if not exists articles_users(id int(11) not null auto_increment, " +
                       "user_id int(11) not null, article_id int(11) not null, primary key(id))")
        cursor.execute("create table if not exists words_articles(id int(11) not null auto_increment, "
                       "article_id int(11) not null, count int(8) not null,\
                       word_id int(11) not null, primary key(id))")
        cursor.execute("create table if not exists citations(id int(11) not null auto_increment, " +
                       "article_id int(11) not null, cited_article_id int(11) not null, primary key(id))")
        cursor.execute("create table if not exists words(id int(11) not null, word varchar(55), primary key(id))")

    @staticmethod
    def drop_database():
        """
        Drop the database.
        """
        print("*** Dropping database ***")
        db = DataParser.get_connection()
        cursor = db.cursor()
        config = DataParser.get_config()
        cursor.execute("drop database if exists %s;" % config['database']['database_name'])
        DataParser.clean_up(db, cursor)

    @staticmethod
    def clean_up(db, cursor):
        """
        Method is always called to make sure database connections are closed and committed
        """
        db.commit()
        cursor.close()
        db.close()
