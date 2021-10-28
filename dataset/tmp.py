    """Returns the eopatches in a way that allows for
    """
def get_eopatches_dataframe(eopatches):
    vector_timeless= []
    vector_timeless = pd.DataFrame(columns = np.concatenate((eopatches[-1].vector_timeless["LOCATION"].columns.values.copy(),["EOPATCH_ID"]))).reset_index(drop=True)
    for i, eopatch in enumerate(eopatches):
        _tmp = eopatch.vector_timeless["LOCATION"].copy(deep=True)
        _tmp["EOPATCH_ID"] = i
        vector_timeless = vector_timeless.append(_tmp)
    return vector_timeless