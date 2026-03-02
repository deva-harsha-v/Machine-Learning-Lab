# ===== DATASET =====
data = [
    (0,0,0,'c0'),
    (1,0,1,'c1'),
    (1,0,0,'c0'),
    (1,1,1,'c1'),
    (0,1,1,'c1'),
    (0,1,1,'c0')
]

test = [1,0,1]

classes=['c0','c1']
total=len(data)

# priors
priors={}
for c in classes:
    priors[c]=sum(1 for d in data if d[3]==c)/total

def get_rows(cls,conditions):
    rows=[d for d in data if d[3]==cls]
    for i,v in conditions:
        rows=[r for r in rows if r[i]==v]
    return rows

# =========================================================
print("=== FULL BAYESIAN (MLE) ===\n")
print("Test Sample:",test,"\n")

post_mle={}

for c in classes:
    print("Class =",c)
    print("Prior P({}) = {}".format(c,priors[c]))
    
    likelihood=1
    conditions=[]
    
    for i,val in enumerate(test):
        rows_class=[d for d in data if d[3]==c]
        filtered=get_rows(c,conditions)
        
        total_rows=len(filtered) if filtered else len(rows_class)
        count=len([r for r in filtered if r[i]==val]) if filtered else len([r for r in rows_class if r[i]==val])
        
        prob = count/total_rows if total_rows>0 else 0
        
        print("P(F{}={} | previous features, {}) = {} (count={}, total_rows={})"
              .format(i+1,val,c,prob,count,total_rows))
        
        likelihood*=prob
        conditions.append((i,val))
    
    posterior=likelihood*priors[c]
    post_mle[c]=posterior
    
    print("Likelihood =",likelihood)
    print("Posterior (unnormalized) =",posterior,"\n")

print("Final Prediction =", max(post_mle,key=post_mle.get))

# =========================================================
print("\n=== BAYESIAN ESTIMATION (LAPLACE) ===\n")
print("Test Sample:",test,"\n")

post_bayes={}

for c in classes:
    print("Class =",c)
    print("Prior P({}) = {}".format(c,priors[c]))
    
    likelihood=1
    conditions=[]
    
    for i,val in enumerate(test):
        rows_class=[d for d in data if d[3]==c]
        filtered=get_rows(c,conditions)
        
        total_rows=len(filtered) if filtered else len(rows_class)
        count=len([r for r in filtered if r[i]==val]) if filtered else len([r for r in rows_class if r[i]==val])
        
        # Laplace smoothing
        prob=(count+1)/(total_rows+2)
        
        print("P(F{}={} | previous features, {}) = {} (count={}, total_rows={})"
              .format(i+1,val,c,prob,count,total_rows))
        
        likelihood*=prob
        conditions.append((i,val))
    
    posterior=likelihood*priors[c]
    post_bayes[c]=posterior
    
    print("Likelihood =",likelihood)
    print("Posterior (unnormalized) =",posterior,"\n")

print("Final Prediction =", max(post_bayes,key=post_bayes.get))
