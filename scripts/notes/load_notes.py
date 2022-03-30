import pickle 

def load_embedding(args):
    # load the pretrained (fasttext) word-embedding
    if not isfile(args.embedding_dir + args.embedding_name.replace('.vec', '.pickle')):
        # Load word embeddings
        embeddings_index = {}
        f = codecs.open(embedding_dir + embedding_name, encoding='utf-8')
        for line in f:
                values = line.rstrip().rsplit(' ')
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        f.close()
        with open(args.embedding_dir + args.embedding_name.replace('.vec', '.pickle'), 'wb') as handle:
            pickle.dump(embeddings_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(args.embedding_dir + args.embedding_name.replace('.vec', '.pickle'), 'rb') as handle:
            embeddings_index = pickle.load(handle)
    return embeddings_index

def load_relevant_metadata(args):
    # Load metadata file
    metadata = pd.read_table(args.notes_metadata_path, 
                            parse_dates = ['admdatetime'],sep='\t')
    
    # select only the relevant metadata that respects the criteria
    metadata_relevant = metadata[metadata.notes_before >= args.min_notes_before]
    metadata_relevant = metadata[metadata.notes_during >= args.min_notes_during]
    metadata_relevant = metadata_relevant[metadata_relevant.losmin >= (args.baseline_hour + args.window_hours * (args.window_list[window_idx-1]))*60] 
    return metadata_relevant


def read_notes(data_path, data_name, header_path):
    # Load clinical notes
    headers = list(pd.read_table(header_path, encoding='utf_8_sig', nrows=1).columns.values)
    usecols = ['PID', 'TIMESTAMP', 'JOURNAL_TEXT']
    return pd.read_table(data_path + data_name, 
                                                sep='\t',
                                                encoding='utf_8_sig',
                                                usecols = usecols,
                                                parse_dates = ['TIMESTAMP'],
                                                names = headers)

def merge_notes(notes, metadata_with_notes):
    # Add metadata info to notes and concatenate the notes to one string.
    notes = pd.merge(notes, metadata_with_notes, left_on = 'PID', right_on = 'enc_cpr', how = 'left')
    notes = notes[notes.TIMESTAMP <= (notes.admdatetime + datetime.timedelta(hours=args.baseline_hour))]
    notes = notes.sort_values(['PID', 'TIMESTAMP'], ascending= [True, True]) # WHICH ORDER TO SORT?
    # concatenate notes
    notes_concat = notes.groupby('courseid_unique')['JOURNAL_TEXT'].apply(lambda x: "%s" % ' THISISTHEENDOFANOTE '.join(x))
    df_notes_concat = pd.DataFrame({'courseid_unique':notes_concat.index, 'JOURNAL_TEXT':notes_concat.values})
    return pd.merge(metadata_with_notes, df_notes_concat, on='courseid_unique')

def train_test_split_notes(notes_merged, ssn_train, ssn_test):
    train_df = notes_merged[notes_merged.enc_cpr.isin(ssn_train)]
    test_df = notes_merged[notes_merged.enc_cpr.isin(ssn_test)]
    assert (train_df.shape[0] + test_df.shape[0] == notes_merged.shape[0]), "Something went wrong with the split!"
    label_names = train_df.dead90.unique().tolist()
    y_train = train_df['dead90'].values
    y_test = test_df['dead90'].values
    raw_docs_train = train_df['JOURNAL_TEXT'].tolist()
    raw_docs_test = test_df['JOURNAL_TEXT'].tolist() 
    
    # Find 'max_seq_len' 
    train_df['doc_len'] = train_df['JOURNAL_TEXT'].apply(lambda words: len(words.split(" "))).fillna(0)
    max_seq_len = np.round(train_df['doc_len'].mean() + train_df['doc_len'].std()).astype(int)
    print('max_seq_len:', max_seq_len)
    return raw_docs_train, raw_docs_test, y_train, y_test, max_seq_len

def preproc_text(raw_journals):
    # Word-tokenize the text and remove stop words.
    processed_docs_train = []
    for doc in raw_journals:
        if USE_REGEX_TOKENIZER:
            tokens = word_tokenizer.tokenize(doc) #OBS: Word-tokenizer
        else:
            tokens = SpaceTokenizer(doc) 
        filtered = [word for word in tokens if word not in stop_words]
        processed_docs_train.append(" ".join(filtered))

    return processed_docs_train  

def tokenize_text(processed_journals,  
                num_words = MAX_NB_WORDS,
                max_seq_len = MAX_SEQ_LEN,
                filters = ',.t', 
                lower = False, 
                char_level = False,
                padding = 'pre', 
                truncating = 'pre'):
    # OBS: Keras Tokenizer by default removes all special characters!!
    tokenizer = Tokenizer(num_words=num_words,
                                                filters=',.\t\n',
                                                lower=lower, 
                                                char_level=char_level) # OBS: Keras Tokenizer
    tokenizer.fit_on_texts(processed_journals)  
    
    word_seq = tokenizer.texts_to_sequences(processed_journals)
    word_index = tokenizer.word_index
    print("dictionary size: ", len(word_index))
    
    # Pad sequences
    #word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len, padding=padding, truncating=truncating)
    #word_seq_test = sequence.pad_sequences(word_seq_test, maxlen=max_seq_len, padding=padding, truncating=truncating)
    
    return word_seq, word_index

def prepare_embed_matrix(word_index, MAX_NB_WORDS, embeddings_index, EMBED_DIM):
    #embedding matrix
    words_not_found = []
    nb_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((nb_words, EMBED_DIM))
    for word, i in word_index.items():
            if i >= nb_words:
                    continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is None:
                    embedding_vector = embeddings_index.get(word.lower())
            if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector
            else:
                    words_not_found.append(word)
    print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    return embedding_matrix, nb_words, words_not_found 
