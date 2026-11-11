# DS Service Payload V1

## 1. Request Payload

**Direction:** DEV → DS Service

```json
{
  "userRequest": {
    "user_id": "user_abc123",
    "session_id": "1ade",
    "query": "Show me top categories by revenue in January 2026 as a chart",
    "mode": "DEFAULT",
    "answer_id": "null"
  }
}
```

---

# 2. Response Payload

**Direction:** DS Service → DEV

```json
{
  "responses": {
    "Greeting Response": {
      "answerId": "ans_20260724_162200_001",
      "type": "text",
      "meta": {
        "dataRefreshedAt": "2026-07-13T00:00:00Z"
      },
      "data": {
        "text": "Hi! I'm your Analytics Assistant. I can help you analyze your financial and transaction data.",
        "table": null,
        "chart": null
      },
      "followUps": null
    },

    "Clarification Response": {
      "answerId": "ans_20260724_162200_001",
      "type": "text",
      "meta": {
        "dataRefreshedAt": "2026-07-13T00:00:00Z"
      },
      "data": {
        "text": "Could you please specify the time period you would like to analyze? For example, provide a month, date range, or year for the revenue analysis.",
        "table": null,
        "chart": null
      },
      "followUps": null
    },

    "Text Response": {
      "answerId": "ans_20260724_162200_001",
      "type": "text",
      "meta": {
        "dataRefreshedAt": "2026-07-13T00:00:00Z"
      },
      "data": {
        "text": "The total revenue for March 2026 was <b>LKR 14,074,053.47</b>, showing the overall revenue generated across all WPOS merchants during the month.",
        "table": null,
        "chart": null
      },
      "followUps": [
        "What was the transaction count for the top performing industry in March 2026?",
        "How does the revenue of the hotel & restaurant industry compare to the previous month?"
      ]
    },

    "Multiselect Response": {
      "answerId": "ans_20260724_162230_005",
      "type": "multi_select",
      "meta": {
        "dataRefreshedAt": "2026-07-13T00:00:00Z"
      },
      "followUps": [],
      "data": {
        "question": "Select the devices you need revenue figures for Carnage.",
        "options": [
          {
            "id": "carnage-1",
            "label": "Carnage 1"
          },
          {
            "id": "carnage-2",
            "label": "Carnage 2"
          },
          {
            "id": "carnage-3",
            "label": "Carnage 3"
          },
          {
            "id": "carnage-local",
            "label": "Carnage Local"
          }
        ]
      }
    },

    "Table Response table_1": {
      "answerId": "ans_20260724_162215_002",
      "type": "table",
      "summary": "The top performing industries in January 2026 based on revenue were Hotel & Restaurant with LKR 4,497,028.23, followed by Textile with LKR 2,613,040.30 and Retail, Supermarkets & Groceries with LKR 1,505,693.31.",
      "meta": {
        "dataRefreshedAt": "2026-07-13T00:00:00Z"
      },
      "followUps": [
        "Show transaction count for these industries",
        "Compare January 2026 revenue with December 2025"
      ],
      "data": {
        "table": {
          "columns": [
            {
              "key": "industry",
              "label": "Industry"
            },
            {
              "key": "revenue",
              "label": "Revenue (LKR)"
            }
          ],
          "rows": [
            {
              "industry": "Hotel & Restaurant",
              "revenue": 4497028.23
            },
            {
              "industry": "Textile",
              "revenue": 2613040.3
            },
            {
              "industry": "Retail, Supermarkets & Groceries",
              "revenue": 1505693.31
            },
            {
              "industry": "Healthcare",
              "revenue": 677562.23
            },
            {
              "industry": "Others",
              "revenue": 664260.34
            }
          ],
          "page": 1,
          "pageSize": 5,
          "totalRows": 5
        }
      }
    },

    "Table Response table_2": {
      "answerId": "ans_20260724_162215_003",
      "type": "table",
      "summary": "Your query returned 3,458 records.",
      "meta": {
        "dataRefreshedAt": "2026-07-13T00:00:00Z"
      },
      "followUps": ["Filter by industry", "Export results"],
      "data": {
        "table": {
          "columns": [
            {
              "key": "merchant",
              "label": "Merchant"
            },
            {
              "key": "industry",
              "label": "Industry"
            },
            {
              "key": "district",
              "label": "District"
            },
            {
              "key": "revenue",
              "label": "Revenue (LKR)"
            }
          ],
          "rows": [
            {
              "merchant": "ABC Retail",
              "industry": "Retail",
              "district": "Colombo",
              "revenue": 2140000
            }
          ],
          "pagination": {
            "page": 1,
            "pageSize": 5,
            "totalRows": 3458,
            "totalPages": 173,
            "hasNext": true,
            "hasPrevious": false
          }
        }
      }
    },

    "Pie Chart Response": {
      "answerId": "ans_20260724_162215_003",
      "type": "pie_chart",
      "title": "Revenue Contribution by Industry",
      "summary": "Retail contributed the highest revenue share in December 2025 with LKR 8,430,220.00, followed by Hospitality with LKR 4,120,400.00. Together, these two industries accounted for the majority of total revenue.",
      "meta": {
        "dataRefreshedAt": "2026-07-13T00:00:00Z"
      },
      "followUps": [
        "Show top merchants in Retail industry",
        "Compare December 2025 revenue with November 2025"
      ],
      "data": {
        "unit": "LKR",
        "segments": [
          {
            "label": "Retail",
            "value": 8430220
          },
          {
            "label": "Hospitality",
            "value": 4120400
          },
          {
            "label": "Services",
            "value": 1850110
          },
          {
            "label": "Healthcare",
            "value": 980500
          },
          {
            "label": "Others",
            "value": 620300
          }
        ],
        "table": {
          "columns": [
            {
              "key": "industry",
              "label": "Industry"
            },
            {
              "key": "revenue",
              "label": "Revenue (LKR)"
            }
          ],
          "rows": [],
          "page": 1,
          "pageSize": 5,
          "totalRows": 127
        }
      }
    },

    "Bar Chart Response": {
      "answerId": "ans_20260724_162230_004",
      "title": "Top 10 merchants",
      "type": "bar_chart",
      "summary": "Top 10 merchants generated LKR 12.4M during June 2026. Highest contributor: ABC Retail (LKR 2.1M).",
      "meta": {
        "dataRefreshedAt": "2026-07-13T00:00:00Z"
      },
      "followUps": ["Show category breakdown", "Compare with previous month"],
      "data": {
        "unit": "LKR",
        "table": {},
        "chart": {
          "unit": "LKR",
          "bars": [
            {
              "label": "ABC Retail",
              "value": 2140000
            },
            {
              "label": "Global Mart",
              "value": 1850000
            },
            {
              "label": "City Electronics",
              "value": 1420000
            },
            {
              "label": "Health First",
              "value": 1120000
            },
            {
              "label": "Super Mart",
              "value": 980000
            }
          ]
        }
      }
    },

    "Line Chart Response": {
      "answerId": "ans_20260724_162230_006",
      "type": "line_chart",
      "title": "Monthly Revenue Trend (January - June 2026)",
      "summary": "Monthly revenue trend for the top merchants from January to June 2026.",
      "meta": {
        "dataRefreshedAt": "2026-07-13T00:00:00Z"
      },
      "followUps": [],
      "data": {
        "unit": "LKR",
        "points": [
          {
            "label": "JAN",
            "value": 1000000
          },
          {
            "label": "FEB",
            "value": 2000000
          },
          {
            "label": "MAR",
            "value": 1500000
          },
          {
            "label": "APR",
            "value": 3000000
          },
          {
            "label": "MAY",
            "value": 3500000
          },
          {
            "label": "JUN",
            "value": 4000000
          }
        ],
        "table": {}
      }
    }
  }
}
```
