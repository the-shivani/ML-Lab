\# Weather Model Selector



\## Overview



Weather Model Selector is a machine learning project that predicts

which numerical weather prediction model is likely to perform best

for a given location and time.



The system compares four weather prediction models:



\- ECMWF

\- GFS

\- ICON

\- GEM



Historical forecast data is compared with ERA5 reanalysis data to

calculate model errors. These historical performance patterns are

then used to train a Random Forest classifier.



\## Objective



Different numerical weather prediction models can perform

differently depending on location and time.



The objective of this project is to automatically select the model

that is expected to provide the most accurate temperature forecast.



\## Methodology



```text

Historical Forecasts

&#x20;       |

&#x20;       v

ECMWF / GFS / ICON / GEM

&#x20;       |

&#x20;       v

ERA5 Reference Data

&#x20;       |

&#x20;       v

Calculate Forecast Errors

&#x20;       |

&#x20;       v

Determine Best Model

&#x20;       |

&#x20;       v

Create Historical Performance Features

&#x20;       |

&#x20;       v

Random Forest Classifier

&#x20;       |

&#x20;       v

Predict Best Weather Model

