def create_features(df):

    features = df[
        [
            "temperature",
            "pressure",
            "humidity"
        ]
    ].copy()

    return features