// Comprehensive Jupyter Notebook Export with Visualizations

async function exportNotebook(modelId) {
    try {
        logToConsole(`Exporting comprehensive Jupyter notebook for model: ${modelId}`, 'info');
        
        // Fetch model details
        const response = await fetch(`/api/models/${modelId}`);
        const model = await response.json();
        
        // Determine model category
        const isRegression = model.metrics.r2_score !== undefined;
        const isClassification = model.metrics.accuracy !== undefined;
        const isClustering = model.metrics.n_clusters !== undefined;
        const isPCA = model.metrics.n_components !== undefined;
        
        // Build cells array
        const cells = [];
        
        // Title cell
        cells.push({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                `# ML Model: ${model.model_name}\n`,
                `\n`,
                `**Model Type:** ${model.model_type}\n`,
                `**Model Category:** ${isRegression ? 'Regression' : isClassification ? 'Classification' : isClustering ? 'Clustering' : 'Dimensionality Reduction'}\n`,
                `**Created:** ${model.created_at}\n`,
                `**Dataset:** ${model.dataset_name}\n`
            ]
        });
        
        // Performance metrics
        let metricsSource = [`\n`, `## Model Performance\n`, `\n`];
        if (isRegression) {
            metricsSource.push(
                `- **R² Score:** ${model.metrics.r2_score}\n`,
                `- **RMSE:** ${model.metrics.rmse}\n`,
                `- **MAE:** ${model.metrics.mae}\n`,
                `- **MSE:** ${model.metrics.mse || 'N/A'}\n`,
                `- **Adjusted R²:** ${model.metrics.adjusted_r2 || 'N/A'}\n`
            );
        } else if (isClassification) {
            metricsSource.push(
                `- **Accuracy:** ${model.metrics.accuracy}\n`,
                `- **Precision:** ${model.metrics.precision}\n`,
                `- **Recall:** ${model.metrics.recall}\n`,
                `- **F1 Score:** ${model.metrics.f1_score}\n`,
                `- **ROC-AUC:** ${model.metrics.roc_auc || 'N/A'}\n`
            );
        } else if (isClustering) {
            metricsSource.push(
                `- **Number of Clusters:** ${model.metrics.n_clusters}\n`,
                `- **Silhouette Score:** ${model.metrics.silhouette_score || 'N/A'}\n`,
                `- **Davies-Bouldin Score:** ${model.metrics.davies_bouldin_score || 'N/A'}\n`
            );
        } else if (isPCA) {
            metricsSource.push(
                `- **Number of Components:** ${model.metrics.n_components}\n`,
                `- **Total Variance Explained:** ${model.metrics.explained_variance_ratio ? (model.metrics.explained_variance_ratio.reduce((a, b) => a + b, 0) * 100).toFixed(2) + '%' : 'N/A'}\n`
            );
        }
        
        cells.push({
            "cell_type": "markdown",
            "metadata": {},
            "source": metricsSource
        });
        
        // Import libraries
        let imports = [
            "# Import required libraries\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import joblib\n",
            "from sklearn.model_selection import train_test_split\n"
        ];
        
        if (isRegression) {
            imports.push(
                "from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error\n",
                "from sklearn.linear_model import LinearRegression, Ridge, Lasso\n",
                "from sklearn.preprocessing import PolynomialFeatures\n",
                "from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\n",
                "from sklearn.svm import SVR\n"
            );
            if (model.model_type.includes('xgboost')) {
                imports.push("import xgboost as xgb\n");
            }
        } else if (isClassification) {
            imports.push(
                "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve\n",
                "from sklearn.linear_model import LogisticRegression\n",
                "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n",
                "from sklearn.svm import SVC\n"
            );
            if (model.model_type.includes('xgboost')) {
                imports.push("import xgboost as xgb\n");
            }
        } else if (isClustering) {
            imports.push(
                "from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering\n",
                "from sklearn.metrics import silhouette_score, davies_bouldin_score\n"
            );
        } else if (isPCA) {
            imports.push(
                "from sklearn.decomposition import PCA\n"
            );
        }
        
        imports.push(
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "# Set plot style\n",
            "plt.style.use('default')\n",
            "sns.set_palette('husl')\n"
        );
        
        cells.push({
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "source": imports
        });
        
        // Configuration cell for data path
        cells.push({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Configuration\n",
                "\n",
                "Set the path to your dataset file here. Update this path to match your local file location."
            ]
        });
        
        cells.push({
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "source": [
                "# ======================\n",
                "# CONFIGURATION\n",
                "# ======================\n",
                "\n",
                "# Path to your CSV dataset\n",
                "DATA_PATH = 'your_data.csv'\n",
                "\n",
                "# Path to the trained model file\n",
                `MODEL_PATH = '${model.model_id}.pkl'\n`,
                "\n",
                "print(f'Data path: {DATA_PATH}')\n",
                "print(f'Model path: {MODEL_PATH}')"
            ]
        });
        
        // Load model
        cells.push({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                `## Load Trained Model\n`,
                `\n`,
                `Download the model file (\`${model.model_id}.pkl\`) and update the \`MODEL_PATH\` variable in the configuration cell above.`
            ]
        });
        
        cells.push({
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "source": [
                `# Load the trained model\n`,
                `model = joblib.load(MODEL_PATH)\n`,
                `print(f"Model loaded successfully: ${model.model_name}")\n`,
                `print(f"Model type: {type(model).__name__}")`
            ]
        });
        
        // Model configuration
        cells.push({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                `## Model Configuration\n`,
                `\n`,
                isRegression || isClassification ? `**Target Variable:** \`${model.target}\`\n` : '',
                `\n`,
                `**Features:** ${model.features.join(', ')}\n`,
                `\n`,
                isRegression || isClassification ? `**Train/Test Split:** ${(model.train_test_split * 100).toFixed(0)}% / ${((1 - model.train_test_split) * 100).toFixed(0)}%` : ''
            ]
        });
        
        // Load data cell
        cells.push({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                `## Load Your Data\n`,
                `\n`,
                `Load your CSV dataset using the \`DATA_PATH\` configured above.`
            ]
        });
        
        cells.push({
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "source": [
                `# Load your data\n`,
                `df = pd.read_csv(DATA_PATH)\n`,
                `\n`,
                `# Display basic information\n`,
                `print(f"Dataset shape: {df.shape}")\n`,
                `print(f"Columns: {list(df.columns)}")\n`,
                `df.head()`
            ]
        });
        
        // Data Preprocessing Section
        if (model.preprocessing) {
            cells.push({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    `## Data Preprocessing\n`,
                    `\n`,
                    `Apply the same preprocessing steps that were used during model training.`
                ]
            });
            
            let preprocessCode = [`# Data Preprocessing\n`, `\n`];
            
            // Handle duplicates
            if (model.preprocessing.removed_duplicates) {
                preprocessCode.push(
                    `# Remove duplicate rows\n`,
                    `print(f"Rows before removing duplicates: {len(df)}")\n`,
                    `df = df.drop_duplicates()\n`,
                    `print(f"Rows after removing duplicates: {len(df)}")\n`,
                    `print(f"Duplicates removed: ${model.preprocessing.duplicates_removed_count || 0}")\n`,
                    `\n`
                );
            }
            
            // Handle missing values
            if (model.preprocessing.missing_value_strategies && Object.keys(model.preprocessing.missing_value_strategies).length > 0) {
                preprocessCode.push(`# Handle missing values\n`);
                
                for (const [col, info] of Object.entries(model.preprocessing.missing_value_strategies)) {
                    preprocessCode.push(`\n# Column: ${col} (${info.missing_count} missing values)\n`);
                    
                    if (info.strategy === 'drop') {
                        preprocessCode.push(`df = df.dropna(subset=['${col}'])\n`);
                    } else if (info.strategy === 'mean') {
                        preprocessCode.push(`df['${col}'] = df['${col}'].fillna(df['${col}'].mean())\n`);
                    } else if (info.strategy === 'median') {
                        preprocessCode.push(`df['${col}'] = df['${col}'].fillna(df['${col}'].median())\n`);
                    } else if (info.strategy === 'mode') {
                        preprocessCode.push(`df['${col}'] = df['${col}'].fillna(df['${col}'].mode()[0] if not df['${col}'].mode().empty else 0)\n`);
                    } else if (info.strategy === 'zero') {
                        preprocessCode.push(`df['${col}'] = df['${col}'].fillna(0)\n`);
                    }
                }
                preprocessCode.push(`\n`);
            }
            
            // Handle target encoding for classification
            if (model.preprocessing.encoded_columns && Object.keys(model.preprocessing.encoded_columns).length > 0) {
                preprocessCode.push(`# Encode categorical columns\n`);
                preprocessCode.push(`from sklearn.preprocessing import LabelEncoder\n`, `\n`);
                
                for (const [col, info] of Object.entries(model.preprocessing.encoded_columns)) {
                    if (info.type === 'target') {
                        preprocessCode.push(
                            `# Encode target variable: ${col}\n`,
                            `le_${col} = LabelEncoder()\n`,
                            `le_${col}.classes_ = np.array(${JSON.stringify(info.classes)})\n`,
                            `# Note: Original classes were: ${JSON.stringify(info.original_values)}\n`,
                            `\n`
                        );
                    }
                }
            }
            
            // Report dropped columns
            if (model.preprocessing.dropped_columns && model.preprocessing.dropped_columns.length > 0) {
                preprocessCode.push(
                    `# Note: The following columns were dropped during training (${model.preprocessing.drop_reason || 'non-numeric'}):\n`,
                    `# ${model.preprocessing.dropped_columns.join(', ')}\n`,
                    `\n`
                );
            }
            
            preprocessCode.push(
                `print(f"Preprocessing complete. Final shape: {df.shape}")\n`
            );
            
            cells.push({
                "cell_type": "code",
                "execution_count": null,
                "metadata": {},
                "source": preprocessCode
            });
        }
        
        // Prepare features and target
        cells.push({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                `## Prepare Features and Target\n`,
                `\n`,
                `Extract the features and target variable${isRegression || isClassification ? '' : ' (if applicable)'}.`
            ]
        });
        
        let prepareDataCode = [
            `# Prepare features\n`,
            `X = df[${JSON.stringify(model.features)}]\n`
        ];
        
        if (isRegression || isClassification) {
            prepareDataCode.push(
                `\n`,
                `# Prepare target\n`,
                `y = df['${model.target}']\n`
            );
            
            // Add encoding if needed
            if (model.preprocessing && model.preprocessing.encoded_columns && model.preprocessing.encoded_columns[model.target]) {
                prepareDataCode.push(
                    `\n`,
                    `# Encode target (if needed)\n`,
                    `if y.dtype == 'object' or y.dtype.name == 'category':\n`,
                    `    y = le_${model.target}.transform(y)\n`
                );
            }
        }
        
        prepareDataCode.push(
            `\n`,
            `print(f"Features shape: {X.shape}")\n`
        );
        
        if (isRegression || isClassification) {
            prepareDataCode.push(`print(f"Target shape: {y.shape}")\n`);
        }
        
        cells.push({
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "source": prepareDataCode
        });
        
        // Make predictions
        cells.push({
            "cell_type": "markdown",
            "metadata": {},
            "source": [`## Make Predictions`]
        });
        
        let predictCode = [];
        if (isRegression || isClassification) {
            predictCode = [
                `# Split data\n`,
                `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=${(1 - model.train_test_split).toFixed(2)}, random_state=42)\n`,
                `\n`,
                `# Make predictions\n`,
                `y_pred = model.predict(X_test)\n`,
                `\n`,
                `print(f"Predictions shape: {y_pred.shape}")\n`,
                `print(f"First 5 predictions: {y_pred[:5]}")`
            ];
        } else if (isClustering) {
            predictCode = [
                `# Predict cluster labels\n`,
                `labels = model.predict(X)\n`,
                `\n`,
                `print(f"Number of samples: {len(labels)}")\n`,
                `print(f"Unique clusters: {np.unique(labels)}")\n`,
                `print(f"Cluster distribution: {np.bincount(labels)}")`
            ];
        } else if (isPCA) {
            predictCode = [
                `# Transform data\n`,
                `X_transformed = model.transform(X)\n`,
                `\n`,
                `print(f"Original shape: {X.shape}")\n`,
                `print(f"Transformed shape: {X_transformed.shape}")`
            ];
        }
        
        cells.push({
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "source": predictCode
        });
        
        // Visualizations section
        cells.push({
            "cell_type": "markdown",
            "metadata": {},
            "source": [`## Visualizations`]
        });
        
        // Add visualization code based on model type
        if (isRegression) {
            // Actual vs Predicted
            cells.push({
                "cell_type": "code",
                "execution_count": null,
                "metadata": {},
                "source": [
                    `# Actual vs Predicted Plot\n`,
                    `plt.figure(figsize=(10, 6))\n`,
                    `plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')\n`,
                    `plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)\n`,
                    `plt.xlabel('Actual Values', fontsize=12)\n`,
                    `plt.ylabel('Predicted Values', fontsize=12)\n`,
                    `plt.title('Actual vs Predicted Values', fontsize=14)\n`,
                    `plt.grid(True, alpha=0.3)\n`,
                    `plt.tight_layout()\n`,
                    `plt.show()`
                ]
            });
            
            // Residual Plot
            cells.push({
                "cell_type": "code",
                "execution_count": null,
                "metadata": {},
                "source": [
                    `# Residual Plot\n`,
                    `residuals = y_test - y_pred\n`,
                    `plt.figure(figsize=(10, 6))\n`,
                    `plt.scatter(y_pred, residuals, alpha=0.6, edgecolors='k')\n`,
                    `plt.axhline(y=0, color='r', linestyle='--', lw=2)\n`,
                    `plt.xlabel('Predicted Values', fontsize=12)\n`,
                    `plt.ylabel('Residuals', fontsize=12)\n`,
                    `plt.title('Residual Plot', fontsize=14)\n`,
                    `plt.grid(True, alpha=0.3)\n`,
                    `plt.tight_layout()\n`,
                    `plt.show()`
                ]
            });
            
            // Feature Importance
            if (model.coefficients) {
                cells.push({
                    "cell_type": "code",
                    "execution_count": null,
                    "metadata": {},
                    "source": [
                        `# Feature Importance / Coefficients\n`,
                        `if hasattr(model, 'coef_'):\n`,
                        `    coefficients = pd.DataFrame({\n`,
                        `        'Feature': ${JSON.stringify(model.features)},\n`,
                        `        'Coefficient': model.coef_ if len(model.coef_.shape) == 1 else model.coef_[0]\n`,
                        `    }).sort_values('Coefficient', key=abs, ascending=False)\n`,
                        `elif hasattr(model, 'feature_importances_'):\n`,
                        `    coefficients = pd.DataFrame({\n`,
                        `        'Feature': ${JSON.stringify(model.features)},\n`,
                        `        'Importance': model.feature_importances_\n`,
                        `    }).sort_values('Importance', ascending=False)\n`,
                        `\n`,
                        `    plt.figure(figsize=(10, 6))\n`,
                        `    plt.barh(coefficients['Feature'], coefficients.iloc[:, 1])\n`,
                        `    plt.xlabel('Value', fontsize=12)\n`,
                        `    plt.ylabel('Feature', fontsize=12)\n`,
                        `    plt.title('Feature Importance', fontsize=14)\n`,
                        `    plt.tight_layout()\n`,
                        `    plt.show()\n`,
                        `    print(coefficients)`
                    ]
                });
            }
        } else if (isClassification) {
            // Confusion Matrix
            cells.push({
                "cell_type": "code",
                "execution_count": null,
                "metadata": {},
                "source": [
                    `# Confusion Matrix\n`,
                    `from sklearn.metrics import confusion_matrix\n`,
                    `cm = confusion_matrix(y_test, y_pred)\n`,
                    `\n`,
                    `plt.figure(figsize=(8, 6))\n`,
                    `sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)\n`,
                    `plt.xlabel('Predicted', fontsize=12)\n`,
                    `plt.ylabel('Actual', fontsize=12)\n`,
                    `plt.title('Confusion Matrix', fontsize=14)\n`,
                    `plt.tight_layout()\n`,
                    `plt.show()\n`,
                    `\n`,
                    `print("Classification Report:")\n`,
                    `print(classification_report(y_test, y_pred))`
                ]
            });
            
            // ROC Curve (for binary classification)
            cells.push({
                "cell_type": "code",
                "execution_count": null,
                "metadata": {},
                "source": [
                    `# ROC Curve (for binary classification)\n`,
                    `if hasattr(model, 'predict_proba') and len(np.unique(y)) == 2:\n`,
                    `    y_pred_proba = model.predict_proba(X_test)[:, 1]\n`,
                    `    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)\n`,
                    `    roc_auc = roc_auc_score(y_test, y_pred_proba)\n`,
                    `    \n`,
                    `    plt.figure(figsize=(8, 6))\n`,
                    `    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')\n`,
                    `    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')\n`,
                    `    plt.xlim([0.0, 1.0])\n`,
                    `    plt.ylim([0.0, 1.05])\n`,
                    `    plt.xlabel('False Positive Rate', fontsize=12)\n`,
                    `    plt.ylabel('True Positive Rate', fontsize=12)\n`,
                    `    plt.title('ROC Curve', fontsize=14)\n`,
                    `    plt.legend(loc="lower right")\n`,
                    `    plt.grid(True, alpha=0.3)\n`,
                    `    plt.tight_layout()\n`,
                    `    plt.show()`
                ]
            });
        } else if (isClustering) {
            // Cluster Scatter Plot
            cells.push({
                "cell_type": "code",
                "execution_count": null,
                "metadata": {},
                "source": [
                    `# Cluster Visualization (2D)\n`,
                    `if X.shape[1] >= 2:\n`,
                    `    plt.figure(figsize=(10, 6))\n`,
                    `    scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis', alpha=0.6, edgecolors='k')\n`,
                    `    plt.xlabel(X.columns[0], fontsize=12)\n`,
                    `    plt.ylabel(X.columns[1], fontsize=12)\n`,
                    `    plt.title('Cluster Visualization', fontsize=14)\n`,
                    `    plt.colorbar(scatter, label='Cluster')\n`,
                    `    plt.grid(True, alpha=0.3)\n`,
                    `    plt.tight_layout()\n`,
                    `    plt.show()\n`,
                    `\n`,
                    `# Cluster sizes\n`,
                    `unique, counts = np.unique(labels, return_counts=True)\n`,
                    `plt.figure(figsize=(8, 5))\n`,
                    `plt.bar(unique, counts)\n`,
                    `plt.xlabel('Cluster', fontsize=12)\n`,
                    `plt.ylabel('Number of Points', fontsize=12)\n`,
                    `plt.title('Cluster Distribution', fontsize=14)\n`,
                    `plt.tight_layout()\n`,
                    `plt.show()`
                ]
            });
        } else if (isPCA) {
            // Scree Plot
            cells.push({
                "cell_type": "code",
                "execution_count": null,
                "metadata": {},
                "source": [
                    `# Scree Plot - Explained Variance\n`,
                    `explained_variance = model.explained_variance_ratio_\n`,
                    `cumulative_variance = np.cumsum(explained_variance)\n`,
                    `\n`,
                    `fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n`,
                    `\n`,
                    `# Individual variance\n`,
                    `ax1.bar(range(1, len(explained_variance) + 1), explained_variance)\n`,
                    `ax1.set_xlabel('Principal Component', fontsize=12)\n`,
                    `ax1.set_ylabel('Explained Variance Ratio', fontsize=12)\n`,
                    `ax1.set_title('Scree Plot', fontsize=14)\n`,
                    `ax1.grid(True, alpha=0.3)\n`,
                    `\n`,
                    `# Cumulative variance\n`,
                    `ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')\n`,
                    `ax2.axhline(y=0.95, color='r', linestyle='--', label='95% variance')\n`,
                    `ax2.set_xlabel('Number of Components', fontsize=12)\n`,
                    `ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)\n`,
                    `ax2.set_title('Cumulative Explained Variance', fontsize=14)\n`,
                    `ax2.legend()\n`,
                    `ax2.grid(True, alpha=0.3)\n`,
                    `\n`,
                    `plt.tight_layout()\n`,
                    `plt.show()\n`,
                    `\n`,
                    `print(f"Total variance explained: {cumulative_variance[-1]*100:.2f}%")`
                ]
            });
            
            // Transformed Data Visualization
            cells.push({
                "cell_type": "code",
                "execution_count": null,
                "metadata": {},
                "source": [
                    `# Visualize Transformed Data (First 2 Components)\n`,
                    `if X_transformed.shape[1] >= 2:\n`,
                    `    plt.figure(figsize=(10, 6))\n`,
                    `    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.6, edgecolors='k')\n`,
                    `    plt.xlabel('First Principal Component', fontsize=12)\n`,
                    `    plt.ylabel('Second Principal Component', fontsize=12)\n`,
                    `    plt.title('PCA - Transformed Data', fontsize=14)\n`,
                    `    plt.grid(True, alpha=0.3)\n`,
                    `    plt.tight_layout()\n`,
                    `    plt.show()`
                ]
            });
        }
        
        // Create notebook object
        const notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.8.0",
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    }
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        };
        
        // Create and download the notebook file
        const blob = new Blob([JSON.stringify(notebook, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${model.model_name}_comprehensive.ipynb`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        logToConsole(`Comprehensive notebook exported: ${model.model_name}_comprehensive.ipynb`, 'success');
        alert(`Notebook exported successfully!\n\nFile: ${model.model_name}_comprehensive.ipynb\n\nIncludes: Complete code, visualizations, and metrics for ${isRegression ? 'regression' : isClassification ? 'classification' : isClustering ? 'clustering' : 'PCA'} model.`);
    } catch (error) {
        alert(`Error exporting notebook: ${error.message}`);
        logToConsole(`Error exporting notebook: ${error.message}`, 'error');
    }
}

