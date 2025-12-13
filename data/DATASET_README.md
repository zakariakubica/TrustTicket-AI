\# TrustTicket Dataset Documentation



\## Overview

This dataset contains 30 labeled ticket listings used for training the TrustTicket AI scam detection system.



\## Dataset Statistics

\- \*\*Total Listings\*\*: 30

\- \*\*Legitimate\*\*: 20 (67%)

\- \*\*Scams\*\*: 10 (33%)

\- \*\*Date Created\*\*: December 2025



\## Features



\### Core Fields

1\. \*\*listing\_id\*\*: Unique identifier (TL001-TL030)

2\. \*\*event\_name\*\*: Concert/event name

3\. \*\*artist\*\*: Performing artist

4\. \*\*venue\*\*: Event location

5\. \*\*date\*\*: Event date

6\. \*\*price\*\*: Ticket price (£)

7\. \*\*platform\*\*: Listing platform (StubHub, Viagogo, Facebook, etc.)

8\. \*\*seller\_rating\*\*: Seller rating (0-5 stars, null for unrated)

9\. \*\*listing\_age\_days\*\*: Days since listing posted

10\. \*\*description\*\*: Listing description text

11\. \*\*label\*\*: Ground truth (legitimate/scam)



\### Derived Features

12\. \*\*weighting\*\*: Sample weight for training (scams: 0.3-1.5, legitimate: 2.3-4.2)

13\. \*\*platform\_trust\_score\*\*: Historical platform reliability (0-1)

14\. \*\*red\_flags\*\*: Comma-separated suspicious indicators



\## Red Flags Included

\- "too\_good\_price" - Price significantly below market

\- "poor\_grammar" - Spelling/grammar errors

\- "urgency\_tactics" - Pressure to buy quickly

\- "payment\_requests" - Unusual payment methods

\- "no\_seller\_history" - New/unrated seller

\- "vague\_details" - Lack of specific information

\- "suspicious\_contact" - Request to contact off-platform



\## Usage

```python

import pandas as pd



\# Load dataset

df = pd.read\_csv('data/raw/ticket\_listings\_dataset.csv')



\# Split features and labels

X = df.drop(\['label', 'listing\_id', 'weighting'], axis=1)

y = df\['label']

weights = df\['weighting']



\# Use weights in training

model.fit(X, y, sample\_weight=weights)

```



\## Data Quality

\- All listings manually labeled

\- Realistic price distributions

\- Representative platform mix

\- Authentic red flag patterns



\## Expansion Plans

\- Week 2: Expand to 200+ listings via web scraping

\- Week 3: Add historical data from real platforms

\- Week 4: Final dataset of 500+ samples



\## File Location

`data/raw/ticket\_listings\_dataset.csv`

