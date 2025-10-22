
export interface Transaction {
  index: number;
  trans_date_trans_time: string;
  merchant: string;
  category: string;
  amt: number;
  city: string;
  state: string;
  lat: number;
  long: number;
  city_pop: number;
  job: string;
  dob: string;
  trans_num: string;
  merch_lat: number;
  merch_long: number;
  is_fraud: number;
}

export interface FraudPrediction {
    fraudPercentage: number;
    analysis: string;
}

export interface GeminiAnalysis {
    originalData: FraudPrediction;
    editedData: FraudPrediction;
}
